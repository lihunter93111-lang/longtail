"""Export hash-verified weights, fresh confirmation inputs and frozen decisions outside Git."""
from common import *
import shutil
import zipfile

def main():
    bundle=ROOT/'handoff_bundle';bundle.mkdir(exist_ok=True)
    mapping={};records=[]
    def copy(source,relative):
        source=Path(source);dest=bundle/relative;dest.parent.mkdir(parents=True,exist_ok=True)
        if not dest.exists():shutil.copy2(source,dest)
        assert sha(source)==sha(dest)
        mapping[str(source)]=str(relative).replace('\\','/')
        return dest
    for split in CONFIG['splits']:
        for decision in sorted((ROOT/split/'decision_models').glob('*_23*.json')):
            state=read(decision)
            for key,digest in [('model_path','model_sha256'),('reference_path','reference_sha256')]:
                if key in state:
                    assert sha(state[key])==state[digest]
                    dest=copy(state[key],Path('weights')/(state[digest]+'.pth'))
                    # Preserve the exact JSON key spelling, including slash conventions.
                    mapping[state[key]]=dest.relative_to(bundle).as_posix()
            copy(decision,Path('decisions')/split/decision.name)
        manifest=ROOT/split/'confirmation/manifest.json';data=read(manifest)
        dest=copy(data['path'],Path('data')/split/'confirmation.mat')
        mapping[data['path']]=dest.relative_to(bundle).as_posix()
        copy(manifest,Path('data')/split/'manifest.json')
        copy(ROOT/split/'confirmation/preflight.json',Path('data')/split/'preflight.json')
        for path in (ROOT/split/'confirmation_results').glob('*.npz'):
            copy(path,Path('predictions')/split/path.name)
        copy(ROOT/split/'confirmation_results/summary.json',Path('predictions')/split/'summary.json')
    write(bundle/'path_map.json',mapping)
    note='''2026-11-11 协作复放包

内容：40 个骨干权重（按 SHA 去重）、100 个冻结决策、两套新确认 MAT 和索引、原始逐样本 logits。没有训练候选池或恢复优化器状态。不要用已消费的确认集继续调模型。
需要配套 Git 仓库 lihunter93111-lang/longtail 的 codex/data-marginal-20261111 分支。
推理入口位于 experiment-log/2026-11-11-01-label-marginal-controls/portable_inference.py。
该入口只导入当前仓库 cjlt，不要求原 E 盘源码目录。path_map.json 内的相对位置以该 JSON 所在文件夹为根解析。
示例（将 BUNDLE 和 REPO 换成实际路径；输出选择包外的新文件）：
python REPO/experiment-log/2026-11-11-01-label-marginal-controls/portable_inference.py --decision BUNDLE/decisions/split1/combo_affine_2300.json --manifest BUNDLE/data/split1/manifest.json --path-map BUNDLE/path_map.json --device cuda --output NEW_OUTPUT.npz
输入为 MAT iq_complex，按每个样本的复数 RMS 标准化后形状 N,2,3000；标签顺序 FMB,AMA,SawFM,SinFM,MFT,ISRJ；sigmoid(logit)>=对应阈值。
阈值和仿射系数已在独立 calibration 与 fusion_fit 拟合。不得在这里重新拟合。
bundle_manifest.json 给出逐文件 SHA256；压缩包自身 SHA 在包外 handoff_bundle_receipt.json。
'''
    (bundle/'README.txt').write_text(note,encoding='utf-8')
    for path in sorted(bundle.rglob('*')):
        if path.is_file() and path.name!='bundle_manifest.json':records.append({'path':path.relative_to(bundle).as_posix(),'bytes':path.stat().st_size,'sha256':sha(path)})
    write(bundle/'bundle_manifest.json',{'files':records,'time':stamp(),'source_sha256':sha(Path(__file__))})
    archive=ROOT/'20261111_collaborator_replay.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_STORED,allowZip64=True) as z:
        for path in sorted(bundle.rglob('*')):
            if path.is_file():z.write(path,path.relative_to(bundle).as_posix())
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None
    write(ROOT/'handoff_bundle_receipt.json',{'archive':str(archive),'bytes':archive.stat().st_size,'sha256':sha(archive),'files':len(records)+1,
        'models':len(list((bundle/'weights').glob('*.pth'))),'decisions':len(list((bundle/'decisions').rglob('*.json'))),
        'includes_confirmation_data':True,'not_in_git':True,'time':stamp()})
    print('HANDOFF_EXPORTED',archive.stat().st_size,flush=True)

if __name__=='__main__':main()
