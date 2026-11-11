"""Run with python -I in a fresh process, without campaign imports."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import torch

def main():
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);p.add_argument('--fixtures',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    sys.path.insert(0,str(a.bundle));from portable_runtime import PortableRuntime,normalize_iq,digest
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    manifest=json.loads((a.bundle/'manifest.json').read_text(encoding='utf-8'))
    for path,expected in manifest['files'].items():assert digest(a.bundle/path)==expected,path
    index=json.loads((a.fixtures/'index.json').read_text(encoding='utf-8'));inputs={};results=[]
    selection=json.loads((a.fixtures/'selection.json').read_text(encoding='utf-8'))
    assert all(len(set(s['combination_codes']))==63 for s in selection.values())
    for split in ['split1','split2']:
        x=normalize_iq(np.load(a.fixtures/(split+'_raw.npy')));torch.testing.assert_close(x,torch.from_numpy(np.load(a.fixtures/(split+'_normalized.npy'))),rtol=0,atol=0);inputs[split]=x.cuda()
    for row in index:
        model=PortableRuntime(a.bundle,row['record']).cuda()
        with torch.inference_mode():z=model(inputs[row['split']]).cpu().numpy()
        expected=np.load(a.fixtures/row['expected']);np.testing.assert_allclose(z,expected,rtol=1e-6,atol=1e-6)
        results.append({'name':row['name'],'maximum_error':float(abs(z-expected).max())});del model
    cpu_results=[]
    selected={'split1/'+name+'_2200' for name in ['reference','natural_none','combo_global','combo_segment','combo_segment_guarded']}
    selected.add('supervision/split1/reference_affine_2200')
    matrix_names=[r['name'] for r in index if r['name'].startswith('matrix/')]
    if matrix_names:selected.add(matrix_names[0])
    for row in index:
        if row['name'] not in selected:continue
        model=PortableRuntime(a.bundle,row['record'])
        with torch.inference_mode():z=model(inputs[row['split']].cpu()).numpy()
        expected=np.load(a.fixtures/row['expected']);np.testing.assert_allclose(z,expected,rtol=1e-5,atol=5e-5)
        cutoff=torch.logit(model.thresholds).numpy();np.testing.assert_array_equal(z>=cutoff,expected>=cutoff)
        cpu_results.append({'name':row['name'],'maximum_error':float(abs(z-expected).max()),'decisions_equal':True});del model
    imports={name:str(Path(module.__file__).resolve()) for name,module in sys.modules.items() if name.startswith('cjlt') and getattr(module,'__file__',None)}
    assert all(Path(path).is_relative_to(a.bundle.resolve()) for path in imports.values())
    result={'passed':True,'isolated_python_process':True,'raw_preprocessing_exact':True,'records':results,'cpu_representative_records':cpu_results,
            'cpu_tolerance':{'rtol':1e-5,'atol':5e-5},'replay_selection':selection,'source_imports':imports,'bundle_manifest_sha256':digest(a.bundle/'manifest.json')}
    a.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print('PORTABLE_VERIFIED',len(results),flush=True)

if __name__=='__main__':main()
