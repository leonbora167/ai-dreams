from pathlib import Path
from tqdm import tqdm
from .tracklets import load_tracklets
from .tracking import track_video
from .features import FeatureManager
from .association import associate


class Pipeline:
    def __init__(self, cfg, videos, log=print): self.cfg, self.videos, self.log=cfg,videos,log
    def run(self):
        root=Path(self.cfg['system']['output_dir'])/'tracks'; root.mkdir(parents=True, exist_ok=True)
        all_tracks=[]
        self.log('Stage 1/4: local detection and ByteTrack tracking')
        for cam,path in self.videos.items():
            cache=root/f'{cam}.json'
            if cache.exists():
                self.log(f'  {cam}: loading cached tracklets from {cache}')
                tracks=load_tracklets(cache)
            else:
                self.log(f'  {cam}: running detector/tracker on {path}')
                tracks=track_video(cam,path,self.cfg,cache,log=self.log)
            self.log(f'  {cam}: {len(tracks)} local tracklet(s)')
            all_tracks.extend(tracks)
        self.log('Stage 2/4: extracting enabled features')
        manager=FeatureManager(self.cfg); descriptors=[]
        with tqdm(total=len(all_tracks), desc='feature extraction', unit='tracklet') as bar:
            for track in all_tracks:
                descriptors.append(manager.extract(track)); bar.update(1)
        self.log('Stage 3/4: associating global identities')
        mapping=associate(all_tracks, descriptors, self.cfg)
        self.log(f'  associated {len(all_tracks)} tracklets into {len(set(mapping.values()))} global identities')
        self.log('Stage 4/4: rendering visualization')
        from .visualization import render
        render(self.videos, mapping, all_tracks, self.cfg)
        self.log(f'Complete. Visualization: {self.cfg["output"]["video"]}')
