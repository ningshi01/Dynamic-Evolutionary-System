import pymilvus

MILVUS_HOST = "192.168.5.76"       # 替换为你的节点 IP
MILVUS_PORT = "32009"   # 替换为 Milvus 的 NodePort

try:
    print(f"Connecting to Milvus at {MILVUS_HOST}:{MILVUS_PORT}...")
    pymilvus.connections.connect(alias='default', host=MILVUS_HOST, port=str(MILVUS_PORT))
    print("Successfully connected to Milvus!")
    print(f"Existing collections: {pymilvus.utility.list_collections()}")
except Exception as e:
    print(f"An error occurred: {e}")
finally:
    if 'default' in pymilvus.connections.list_connections():
        pymilvus.connections.disconnect('default')
        print("Disconnected from Milvus.")