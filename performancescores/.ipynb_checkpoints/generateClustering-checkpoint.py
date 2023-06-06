import networkit as nk

G = nk.readGraph("/work/global/dyane-instances/com-youtube", nk.Format.SNAP)
par = nk.community.ParallelLeiden(G, iterations=32).run().getPartition()
par2 = nk.community.PLM(G, refine=True, maxIter=32).run().getPartition()

print("Leiden mod: ", 


