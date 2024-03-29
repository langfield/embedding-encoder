import numpy as np

def rand_embedding_generator(vectors):
   
    shape = vectors.shape
    inputs = shape[0]
    dims = shape[1]
 
    # shape [<inputs>,<dimensions>]
    rand_emb_array = []

    for i in range(inputs):
        
        # Shape: [<dims>]
        vec = np.random.rand(dims)
        vec = vec / np.linalg.norm(vec)
        rand_emb_array.append(vec)

    print(vec)

#========1=========2=========3=========4=========5=========6=========7==

if __name__ == "__main__":
    # stuff only to run when not called via 'import' here
    
    dims = 100
    inputs = 1000

    vectors = np.random.rand(1000, 100)
    
    # Generate. 
    rand_embedding_generator(vectors)

