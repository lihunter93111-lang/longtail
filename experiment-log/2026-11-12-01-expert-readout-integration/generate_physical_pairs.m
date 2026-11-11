function generate_physical_pairs(protocolPath,split,role)
    protocol=jsondecode(fileread(protocolPath));
    if strcmp(role,'confirmation')
        frozen=jsondecode(fileread(fullfile(protocol.storage,'final_freeze.json')));assert(frozen.all_models_and_calibrations_frozen);
    end
    data=jsondecode(fileread(fullfile(protocol.storage,split,role,'manifest.json')));
    folder=fullfile(protocol.storage,split,['physical_',role]);receiptPath=fullfile(folder,'generation_receipt.json');
    if isfile(receiptPath),fprintf('PHYSICAL_REUSE %s %s\n',split,role);return;end
    generatorRoot=fullfile(protocol.repository,'data_generation','0716','可配置组合干扰数据生成');
    addpath(generatorRoot,fullfile(generatorRoot,'core'),fullfile(generatorRoot,'generators'));
    original=load(data.path,'config','iq_complex');cfg=original.config;
    registry=build_jamming_registry(cfg);schema=build_parameter_schema(registry);manifest=build_closedset_manifest(cfg,registry,'train');reference=build_reference_signal(cfg);
    labels={'FMB','AMA','SawFM','SinFM','MFT','ISRJ'};names={'gap0','gap10','gap15','absent'};gaps=[0,10,15,NaN];
    n=numel(data.indices);arrays=cell(4,1);
    for g=1:4
        arrays{g}=struct('x_real',zeros(n,3000,'single'),'x_imag',zeros(n,3000,'single'),'positions',zeros(n,1), ...
            'target_labels',zeros(n,1),'labels',zeros(n,6),'relative_power_dB',nan(n,6),'receiver_noise_seed',zeros(n,1),'count',0);
    end
    records=cell(n,1);accepted=0;maxRebuild=0;maxDelta=0;maxRelative=0;
    for position=1:n
        rawIndex=data.indices(position)+1;row=manifest(rawIndex,:);sample=compose_jamming_sample(cfg,registry,schema,row,reference);
        saved=original.iq_complex(rawIndex,:);rebuild=max(abs(sample.iq_complex-saved));assert(rebuild<=1e-9*max(1,max(abs(saved))));maxRebuild=max(maxRebuild,rebuild);
        active=sample.active_type_codes;power=sample.component_effective_JNR_dB;candidates=[];maxima=nan(1,6);
        if numel(active)>1
            for label=1:6
                component=find(strcmp(active,labels{label}));if isempty(component),continue;end
                other=power;other(component)=-Inf;maxima(label)=max(other);
                if maxima(label)>=25,candidates(end+1)=label;end
            end
        end
        if isempty(candidates),continue;end
        label=candidates(mod(sample.sample_seed,numel(candidates))+1);component=find(strcmp(active,labels{label}));
        stream=RandStream('mt19937ar','Seed',sample.component_seeds(component));[clean,params]=generate_single_jamming(labels{label},cfg,registry,stream,reference);
        assert(abs(params.JNR_dB-power(component))<1e-10);
        target=calibrate_component_power(clean,10*log10(cfg.radar.noise_power)+power(component));accepted=accepted+1;
        records{accepted}=struct('position',position-1,'raw_row',rawIndex-1,'sample_seed',sample.sample_seed,'receiver_noise_seed',sample.receiver_noise_seed, ...
            'component_seeds',sample.component_seeds,'target_label',label-1,'original_JNR_dB',power(component),'other_max_JNR_dB',maxima(label),'reconstruction_max_abs',rebuild);
        for g=1:4
            labelValues=double(sample.known_label);effective=power;
            if g==4
                changed=sample.iq_complex-target;expectedDelta=-target;labelValues(label)=0;effective(component)=-Inf;
            else
                newPower=maxima(label)-gaps(g);assert(newPower>=10 && newPower<=30);
                changedRow=row;offsets=zeros(1,numel(power));offsets(component)=newPower-power(component);changedRow.component_target_power_dB{1}=offsets;
                weak=false(1,numel(power));weak(component)=gaps(g)>=10;changedRow.weak_component_mask{1}=weak;
                intervention=compose_jamming_sample(cfg,registry,schema,changedRow,reference);
                assert(isequal(intervention.component_seeds,sample.component_seeds) && intervention.receiver_noise_seed==sample.receiver_noise_seed);
                changed=intervention.iq_complex;expectedDelta=target*(10^((newPower-power(component))/20)-1);effective(component)=newPower;
                assert(max(abs(intervention.component_effective_JNR_dB-effective))<1e-10);
            end
            residual=max(abs(changed-sample.iq_complex-expectedDelta));relative=residual/max(1,max(abs(changed)));assert(relative<1e-12);
            maxDelta=max(maxDelta,residual);maxRelative=max(maxRelative,relative);
            physical=nan(1,6);physical(logical(sample.known_label))=effective;
            if g==4,physical(label)=NaN;end
            rel=physical-max(physical,[],'omitnan');
            arrays{g}.x_real(accepted,:)=single(real(changed));arrays{g}.x_imag(accepted,:)=single(imag(changed));
            arrays{g}.positions(accepted)=position-1;arrays{g}.target_labels(accepted)=label-1;arrays{g}.labels(accepted,:)=labelValues;
            arrays{g}.relative_power_dB(accepted,:)=rel;arrays{g}.receiver_noise_seed(accepted)=sample.receiver_noise_seed;arrays{g}.count=accepted;
        end
        if mod(accepted,500)==0,fprintf('PHYSICAL_PROGRESS %s %s %d\n',split,role,accepted);end
    end
    assert(accepted>0);if ~isfolder(folder),mkdir(folder);end
    for g=1:4
        output=arrays{g};fields=fieldnames(output);
        for f=1:numel(fields),name=fields{f};if ~strcmp(name,'count'),output.(name)=output.(name)(1:accepted,:);end;end
        save(fullfile(folder,[names{g},'.mat']),'-struct','output','-v7.3');
    end
    write_json(fullfile(folder,'identities.json'),vertcat(records{1:accepted}));
    write_json(receiptPath,struct('role',role,'source_path',data.path,'source_sha256',data.sha256,'identities',accepted,'views',{names}, ...
        'all_six_targets_eligible',true,'shared_identities_across_all_views',true,'raw_reconstruction_max_abs',maxRebuild,'delta_max_abs',maxDelta,'delta_max_relative',maxRelative));
    fprintf('PHYSICAL_COMPLETE %s %s %d\n',split,role,accepted);
end
function write_json(path,value)
    fid=fopen(path,'w');assert(fid~=-1);cleaner=onCleanup(@()fclose(fid));fwrite(fid,unicode2native(jsonencode(value,PrettyPrint=true),'UTF-8'),'uint8');
end
