import numpy as np


def _cos(a,b): return float(np.dot(a,b) / ((np.linalg.norm(a)*np.linalg.norm(b))+1e-8))


def associate(items, descriptors, cfg):
    """Return (camera_id, local_id) -> stable global integer IDs."""
    n=len(items)
    if not n: return {}
    fcfg=cfg['features']; threshold=cfg['association']['similarity_threshold']
    adjacency=cfg['cameras']['topology'].get('adjacency', {})
    parent=list(range(n))
    def find(x):
        while parent[x]!=x: parent[x]=parent[parent[x]]; x=parent[x]
        return x
    def union(a,b):
        a,b=find(a),find(b)
        if a!=b: parent[b]=a
    for i in range(n):
        if descriptors[i].get('_skip'): continue
        for j in range(i+1,n):
            if descriptors[j].get('_skip'): continue
            a,b=items[i],items[j]
            if a.camera_id == b.camera_id: continue
            if fcfg.get('topology',{}).get('enabled') and b.camera_id not in adjacency.get(a.camera_id,[]): continue
            gap=max(a.start_time-b.end_time, b.start_time-a.end_time)
            if fcfg.get('temporal',{}).get('enabled'):
                topo=cfg['cameras']['topology'];
                if gap < topo.get('min_transition_seconds',1) or gap > topo.get('max_transition_seconds',30): continue
            scores=[]; weights=[]
            for name, spec in fcfg.items():
                if not spec.get('enabled') or name in ('topology','temporal','quality','geometry','pose'): continue
                if name in descriptors[i] and name in descriptors[j]: scores.append(_cos(descriptors[i][name], descriptors[j][name])); weights.append(spec.get('weight',0))
            if scores and np.average(scores, weights=weights) >= threshold: union(i,j)
    roots={}; result={}
    for i,item in enumerate(items):
        root=find(i); roots.setdefault(root,len(roots)+1); result[(item.camera_id,item.local_id)]=roots[root]
    return result
