"""Measure C1 native-kernel RTL and produce openTPU Lens profiles."""
import argparse,json
from pathlib import Path
from opentpu import c1_hbm_rtl as H,c1_rtl as R,lens

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,default=Path('build/lens/c1/native'));a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True);profiles=[];results=[]
    case=R.workload(n=257,kb=9,ab=0,k0=0)
    for latency,delay in [(10,0),(30,0),(100,0),(30,150)]:
        name=f'c1-native-n257-kb9-lat{latency}-pc0delay{delay}'
        directory=a.out/name
        result=H.run(case,directory,latency=latency,delay_pc0=delay,stall=20,seed=921)
        events=[list(map(int,line.split())) for line in (directory/'events.txt').read_text().splitlines()]
        assert len(events)==result['cycles']
        assert sum(row[2].bit_count() for row in events)==result['counters'][0x48]
        assert sum(row[3].bit_count() for row in events)==result['counters'][0x4c]
        assert sum(row[4] for row in events)==result['counters'][0x40]
        profiles.append(H.lens_data(result,events,name));results.append(dict(name=name,**result))
        print(name,result['cycles'],flush=True)
    lens.save(profiles,a.out/'c1-native-rtl.otpuprof')
    (a.out/'c1-native-rtl.html').write_text(lens.render(profiles,title='C1 native kernel RTL — synthetic memory service'))
    (a.out/'validation.json').write_text(json.dumps(results,indent=2)+'\n')
if __name__=='__main__':main()
