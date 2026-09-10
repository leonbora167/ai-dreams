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


def batch_associate(items, descriptors, cfg):
    """Reconcile all completed tracklets using constrained score-ordered links.

    Greedy online assignments are useful for provisional live IDs, but they can
    never reconsider an early mistake. This final pass uses all tracklets,
    processes strongest cross-camera matches first and blocks overlapping tracks
    from the same camera from entering one identity cluster.
    """
    n = len(items)
    valid = [i for i, d in enumerate(descriptors) if not d.get('_skip')]
    for i in valid: descriptors[i]['_cfg'] = cfg
    threshold = cfg['association']['similarity_threshold']
    adjacency = cfg['cameras']['topology'].get('adjacency', {})
    edges = []

    for i in valid:
        for j in valid:
            if i >= j or items[i].camera_id == items[j].camera_id: continue
            a, b = items[i], items[j]
            if cfg['features'].get('topology', {}).get('enabled') and b.camera_id not in adjacency.get(a.camera_id, []): continue
            gap = max(a.start_time-b.end_time, b.start_time-a.end_time)
            if cfg['features'].get('temporal', {}).get('enabled'):
                topo = cfg['cameras']['topology']
                if gap < topo.get('min_transition_seconds', 1) or gap > topo.get('max_transition_seconds', 30): continue
            score = _descriptor_similarity(descriptors[i], descriptors[j], cfg)
            if score >= threshold:
                edges.append((score, i, j))

    parent = {i: i for i in valid}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    def compatible(a, b):
        for x in valid:
            if find(x) != a: continue
            for y in valid:
                if find(y) == b and items[x].camera_id == items[y].camera_id:
                    if max(items[x].start_time, items[y].start_time) <= min(items[x].end_time, items[y].end_time): return False
        return True
    # Merge strongest links first. Unlike mutual-best matching, this lets
    # several non-overlapping fragments of one person join one identity.
    for score, i, j in sorted(edges, reverse=True):
        ri, rj = find(i), find(j)
        if ri != rj and compatible(ri, rj): parent[rj] = ri

    groups = {}
    for i in valid: groups.setdefault(find(i), []).append(i)
    ordered = sorted(groups.values(), key=lambda g: min(items[i].start_time for i in g))
    result = {}
    for gid, group in enumerate(ordered, 1):
        for i in group: result[(items[i].camera_id, items[i].local_id)] = gid
    for i, item in enumerate(items):
        if (item.camera_id, item.local_id) not in result: result[(item.camera_id, item.local_id)] = max(result.values(), default=0) + 1
    return result


def _descriptor_similarity(a, b, cfg):
    values, weights = [], []
    for name, spec in cfg['features'].items():
        if not spec.get('enabled') or name in ('topology','temporal','quality','geometry','pose'): continue
        if name in a and name in b:
            values.append(_cos(a[name], b[name])); weights.append(spec.get('weight', 0))
    return float(np.average(values, weights=weights)) if values else -1.
