from rag_client import discover_chroma_backends, initialize_rag_system, retrieve_documents
import json
print('DISCOVERING...')
backends = discover_chroma_backends()
print(json.dumps(backends, indent=2))
for k,info in backends.items():
    print('\nBACKEND:', k)
    print('dir:', info.get('directory'))
    try:
        coll = initialize_rag_system(info.get('directory'), info.get('collection_name'))
        print('collection object:', type(coll))
        try:
            cnt = coll.count()
            print('count:', cnt)
        except Exception as e:
            print('count error:', type(e).__name__, e)
        try:
            res = retrieve_documents(coll, 'Apollo 11', n_results=3, mission_filter=None)
            print('retrieve result keys:', list(res.keys()) if res else None)
            print('documents len:', len(res.get('documents',[])) if res else None)
            if res and res.get('documents'):
                doc0 = res['documents'][0] if res['documents'][0] else '<empty>'
                print('first doc preview:', doc0[:300])
        except Exception as e:
            print('retrieve error:', type(e).__name__, e)
    except Exception as e:
        print('init error:', type(e).__name__, e)
