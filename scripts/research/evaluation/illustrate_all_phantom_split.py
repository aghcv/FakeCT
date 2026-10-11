#!/usr/bin/env python3
"""Large-type poster SVG of the proposed all-12-phantom split design."""
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR','/tmp/fakect-all-phantom-split-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'outputs/poster/bmes2026/training-split-all12-r2'
W,H=1600,970


def main():
    if OUT.exists():raise FileExistsError('Use a new diagram revision')
    OUT.mkdir(parents=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','svg.fonttype':'none','pdf.fonttype':42})
    fig=plt.figure(figsize=(W/72,H/72));ax=fig.add_axes([0,0,1,1])
    ax.set_xlim(0,W);ax.set_ylim(H,0);ax.axis('off');fig.patch.set_facecolor('white')
    ink,cyan,magenta,gray='#183144','#009db8','#ce009e','#506677'

    def text(x,y,label,size=32,weight='normal',color=ink,align='left'):
        return ax.text(x,y,label,fontsize=size,fontweight=weight,color=color,
                       ha=align,va='center',zorder=4)

    def box(x,y,w,h,border=ink,fill='#f3f6f8'):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0,rounding_size=18',
            facecolor=fill,edgecolor=border,linewidth=3,zorder=2))

    def arrow(points,color=ink):
        for a,b in zip(points[:-2],points[1:-1]):
            ax.plot([a[0],b[0]],[a[1],b[1]],color=color,linewidth=3,zorder=1)
        ax.add_patch(FancyArrowPatch(points[-2],points[-1],arrowstyle='-|>',
            color=color,linewidth=3,mutation_scale=24,zorder=1))

    text(40,53,'Six-family training and testing',46,'bold')
    text(40,101,'ALL 12 PHANTOMS INCLUDED · PROPOSED DESIGN',27,color=gray)
    box(40,138,1520,150)
    for x in (565,1115):ax.plot([x,x],[160,266],color='#c6d3da',linewidth=2)
    text(302,177,'REFERENCE COHORT',25,'bold',gray,'center')
    text(302,225,'12 phantoms · 6 families',33,'bold',align='center')
    text(840,177,'GENERATE PER PHANTOM',25,'bold',gray,'center')
    text(840,225,'256 siblings + original',33,'bold',align='center')
    text(1337,177,'GENERATION POOL',25,'bold',gray,'center')
    text(1337,225,'3,084 volumes',37,'bold',align='center')

    text(800,332,'ONE ROTATION · KEEP EACH FAMILY TOGETHER',27,'bold',gray,'center')
    arrow([(405,288),(405,384)],cyan)
    arrow([(1195,288),(1195,384)],magenta)
    box(40,391,730,235,cyan,'#e9faff')
    box(830,391,730,235,magenta,'#fff0fb')
    text(75,434,'TRAIN',32,'bold',cyan)
    text(75,478,'5 families · 10 phantoms',32)
    text(75,544,'170',69,'bold',cyan)
    text(265,549,'training volumes',34)
    text(75,593,'10 × (1 original + 16 selected siblings)',29)
    text(865,434,'TEST / EVALUATION',32,'bold',magenta)
    text(865,478,'1 family · 2 unseen phantoms',32)
    text(865,544,'26',69,'bold',magenta)
    text(1000,549,'held-out volumes',34)
    text(865,593,'2 × (1 original + 12 selected siblings)',29)

    arrow([(405,626),(405,659)],cyan)
    arrow([(1195,626),(1195,659)],magenta)
    box(40,668,730,89,cyan,'#e9faff')
    box(830,668,730,89,magenta,'#fff0fb')
    text(405,712,'Fine-tune TotalSegmentator',33,'bold',align='center')
    text(1195,712,'Pretrained / Fine-tuned vs reference',29,'bold',align='center')
    arrow([(770,712),(823,712)],cyan)
    arrow([(1195,757),(1195,791)],magenta)

    box(40,800,1520,110)
    text(75,840,'Rotate the held-out family six times',34,'bold')
    text(75,884,'Average Dice across addition / removal burden',27)
    text(1310,838,'156 test volumes',44,'bold',magenta,'center')
    text(1310,883,'12 originals + 144 siblings',28,align='center')
    text(40,947,'Selected samples: 16 training / 12 test siblings per phantom',24,color=gray)
    text(1560,947,'Test / evaluation share one held-out set',24,color=gray,align='right')

    for suffix in ('svg','pdf','png'):
        fig.savefig(OUT/('training-testing-evaluation-all12.'+suffix),dpi=120,
            facecolor='white',bbox_inches=None)
    plt.close(fig)
    counts={'source_phantoms':12,'families':6,'generated_siblings_per_phantom':256,
            'generation_pool':12*(256+1),'train_families':5,'train_phantoms':10,
            'selected_train_siblings_per_phantom':16,'training_volumes_per_fold':10*(16+1),
            'test_families':1,'test_phantoms':2,'selected_test_siblings_per_phantom':12,
            'test_volumes_per_fold':2*(12+1),'rotations':6,'unique_test_volumes':6*2*(12+1)}
    receipt={'schema_version':'fakect.all12-split-design-figure/1',
        'scope':'Proposed all-12 design assuming the existing 16/12 sibling curation; not completed experiment results',
        'counts':counts,'editable_svg_text':True,'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'output_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.iterdir() if p.is_file()}}
    (OUT/'provenance.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(OUT/'training-testing-evaluation-all12.svg')


if __name__=='__main__':main()
