import torch
import numpy as np
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory
import networkx as nx
import matplotlib.pyplot as plt


# Define a small set of triples for the dummy knowledge graph
triples = [
    ('Alice', 'likes', 'Bob'),
    ('Bob', 'dislikes', 'Charlie'),
    ('Charlie', 'knows', 'Alice'),
    ('David', 'likes', 'Eve'),
    ('Eve', 'dislikes', 'Bob'),
]

# Generate a KG representation using NetworkX
# Create a directed graph
G = nx.DiGraph()

# Add triples to the graph
for head, relation, tail in triples:
    G.add_edge(head, tail, label=relation)

# Draw the graph
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_size=2000, font_size=10, node_color='lightblue', font_weight='bold', font_color='darkred')
edge_labels = nx.get_edge_attributes(G, 'label')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
plt.show()

# Convert the list of triples to a NumPy array
triples_array = np.array(triples)
print("Triples Array:\n", triples_array)

# Create a TriplesFactory from the triples
triples_factory = TriplesFactory.from_labeled_triples(triples_array)
print("Triples Factory:\n", triples_factory)

# Split the triples into training, testing, and validation sets
training, testing, validation = triples_factory.split([0.8, 0.1, 0.1])

# Define the RESCAL model and train it using the pipeline
result = pipeline(
    model='RESCAL',
    training=training,
    testing=testing,
    validation=validation,
    training_kwargs=dict(num_epochs=100, batch_size=2),
    optimizer_kwargs=dict(lr=0.01),
)

# Extract the entity and relation embeddings
entity_embeddings = result.model.entity_representations[0](indices=None).cpu().detach().numpy()
relation_embeddings = result.model.relation_representations[0](indices=None).cpu().detach().numpy()

# Example of how to use embeddings
print("Entity Embeddings:\n", entity_embeddings)
print("Relation Embeddings:\n", relation_embeddings)

# Define GAN components
class Generator(nn.Module):
    def __init__(self, entity_size, embedding_dim):
        super(Generator, self).__init__()
        self.entity_size = entity_size
        self.embedding_dim = embedding_dim
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, entity_size)
        )

    def forward(self, noise):
        return self.fc(noise)

class Discriminator(nn.Module):
    def __init__(self, entity_size, embedding_dim):
        super(Discriminator, self).__init__()
        self.entity_size = entity_size
        self.embedding_dim = embedding_dim
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, embeddings):
        return self.fc(embeddings)

# Hyperparameters
embedding_dim = entity_embeddings.shape[1]
entity_size = len(entity_embeddings)
learning_rate = 0.0002
batch_size = 2
num_epochs = 1000

# Initialize generator and discriminator
generator = Generator(entity_size, embedding_dim)
discriminator = Discriminator(entity_size, embedding_dim)

# Loss function and optimizers
adversarial_loss = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=learning_rate)
optimizer_D = optim.Adam(discriminator.parameters(), lr=learning_rate)

# Training
for epoch in range(num_epochs):
    # Train Discriminator
    optimizer_D.zero_grad()

    # Sample real data
    real_embeddings = torch.tensor(entity_embeddings[np.random.randint(0, entity_embeddings.shape[0], batch_size)], dtype=torch.float)
    real_labels = torch.ones(batch_size, 1)
    fake_labels = torch.zeros(batch_size, 1)

    # Generate fake data
    noise = torch.randn(batch_size, embedding_dim)
    generated_embeddings = generator(noise)

    # Discriminator loss
    real_validity = discriminator(real_embeddings)
    fake_validity = discriminator(generated_embeddings)

    real_loss = adversarial_loss(real_validity, real_labels)
    fake_loss = adversarial_loss(fake_validity, fake_labels)
    d_loss = (real_loss + fake_loss) / 2

    d_loss.backward()
    optimizer_D.step()

    # Train Generator
    optimizer_G.zero_grad()

    # Generate fake data
    noise = torch.randn(batch_size, embedding_dim)
    generated_embeddings = generator(noise)

    # Generator loss
    validity = discriminator(generated_embeddings)
    g_loss = adversarial_loss(validity, real_labels)

    g_loss.backward()
    optimizer_G.step()

    # Print the progress
    if epoch % 100 == 0:
        print(f"[Epoch {epoch}/{num_epochs}] [D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]")
