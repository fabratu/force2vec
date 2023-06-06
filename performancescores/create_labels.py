from dgl.data import FlickrDataset

dataset = FlickrDataset()

g = dataset[0]
labels = g.ndata['label']
print(labels)
print(len(labels))

f = open("flickr.nodes.labels", "w")
for index, value in enumerate(labels):
    f.write(str(index+1))
    f.write("\t")
    f.write(str(value.item()))
    f.write("\n")


