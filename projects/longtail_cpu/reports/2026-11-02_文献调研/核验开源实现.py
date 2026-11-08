import concurrent.futures as cf
import json, urllib.request, pathlib, hashlib

OUT=pathlib.Path('/ultrafast_share/xuwang_group/wxwu/longtail/reports/2026-11-02_文献调研')
OUT.mkdir(parents=True,exist_ok=True)
CACHE=pathlib.Path('/tmp/longtail-research-source-cache');CACHE.mkdir(exist_ok=True)
REPOS={
 'Alibaba-MIIL/ASL':['src/loss_functions/losses.py'],
 'Alibaba-MIIL/ML_Decoder':['src_files/ml_decoder/ml_decoder.py'],
 'SlongLiu/query2labels':['lib/models/query2label.py'],
 'qiuqiangkong/audioset_tagging_cnn':['pytorch/models.py'],
 'YuanGongND/psla':['src/models/Models.py'],
 'wutong16/DistributionBalancedLoss':['mllt/models/losses/resample_loss.py'],
 'Kunmonkey/DR-Loss':['mllt/models/losses/resample_loss.py'],
 'Roche/BalancedLossNLP':['README.md'],
 'facebookresearch/classifier-balancing':['main.py'],
 'naplab/Conv-TasNet':['conv_tasnet.py'],
 'asteroid-team/asteroid':['asteroid/models/conv_tasnet.py'],
 'RFChallenge/icassp2024rfchallenge':['README.md'],
 'google-research/sound-separation':['README.md'],
}

def get(url):
 req=urllib.request.Request(url,headers={'User-Agent':'longtail-literature-audit','Accept':'application/vnd.github+json'})
 with urllib.request.urlopen(req,timeout=25) as f:return f.read()

def work(item):
 repo,paths=item; result={'repository':repo,'url':'https://github.com/'+repo,'checked_date':'2026-11-02','runtime_tested':False}
 try:
  meta=json.loads(get('https://api.github.com/repos/'+repo))
  result.update(default_branch=meta['default_branch'],archived=meta['archived'],license=(meta.get('license')or{}).get('spdx_id','NOT_DETECTED'))
  commits=json.loads(get('https://api.github.com/repos/'+repo+'/commits?per_page=1&until=2026-11-02T23:59:59Z'))
  commit=commits[0];sha=commit['sha'];result.update(commit=sha,commit_date=commit['commit']['committer']['date'])
  files=[]
  for path in list(dict.fromkeys(['README.md']+paths)):
   url=f'https://raw.githubusercontent.com/{repo}/{sha}/{path}'
   try:
    body=get(url);dest=CACHE/(repo.replace('/','__')+'__'+path.replace('/','__'));dest.write_bytes(body)
    txt=body.decode(errors='replace');lines=txt.splitlines()
    cues=[{'line':i,'text':line.strip()[:200]} for i,line in enumerate(lines,1) if any(x in line for x in ['class MLDecoder','class Asymmetric','class Cnn14_DecisionLevelAtt','class Wavegram_Logmel','class AttBlock','class Query2Label','class ResampleLoss','class ConvTasNet','def forward','class DRLoss','def get_criterion','class Conv_TasNet'])]
    files.append({'path':path,'url':f'https://github.com/{repo}/blob/{sha}/{path}','sha256':hashlib.sha256(body).hexdigest(),'lines':len(lines),'local_cache':str(dest),'cues':cues[:24]})
   except Exception as e:files.append({'path':path,'error':str(e)})
  result['files']=files
 except Exception as e:result['error']=repr(e)
 return result

results=[]
with cf.ThreadPoolExecutor(max_workers=6) as pool:
 for r in pool.map(work,REPOS.items()):
  results.append(r)
  print(json.dumps({k:r.get(k) for k in ['repository','license','commit','commit_date','error']}|{'files':[{k:v.get(k) for k in ['path','lines','error']} for v in r.get('files',[])]},ensure_ascii=False),flush=True)
(OUT/'开源实现核验.json').write_text(json.dumps(results,ensure_ascii=False,indent=2))
