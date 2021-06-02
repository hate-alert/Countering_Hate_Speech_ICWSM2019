import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
# module_url = "https://tfhub.dev/google/universal-sentence-encoder/2" #@param ["https://tfhub.dev/google/universal-sentence-encoder/2", "https://tfhub.dev/google/universal-sentence-encoder-large/3"]
# embed = hub.Module(module_url)
# # config = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=12,
# #                        allow_soft_placement=True, device_count = {'CPU': 12})

module_url = "https://tfhub.dev/google/universal-sentence-encoder/4" #@param ["https://tfhub.dev/google/universal-sentence-encoder/4", "https://tfhub.dev/google/universal-sentence-encoder-large/5"]
model = hub.load(module_url)
print ("module %s loaded" % module_url)
def embed(input):
  return model(input)

def get_embeddings(messages):
      
    # with tf.Session() as session:
    #         session.run([tf.global_variables_initializer(), tf.tables_initializer()])
    #         message_emb = session.run(embed(messages))
    message_emb = embed(messages)
    print("ending")
    return np.array(message_emb)