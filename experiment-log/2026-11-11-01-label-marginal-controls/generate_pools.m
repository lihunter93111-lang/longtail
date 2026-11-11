function generate_pools(protocolPath,split,roles)
    protocol = jsondecode(fileread(protocolPath));
    generatorRoot = fullfile(protocol.repository,'data_generation','0716','可配置组合干扰数据生成');
    addpath(generatorRoot,fullfile(generatorRoot,'core'),fullfile(generatorRoot,'generators'));
    original = jsondecode(fileread(protocol.generator_config));
    original.params = row_ranges(original.params);
    for i=1:numel(roles)
        role=roles{i};
        folder=fullfile(protocol.storage,split,'data',role);
        receiptPath=fullfile(folder,'generation_receipt.json');
        if isfile(receiptPath), continue; end
        if strcmp(role,'confirmation_pool')
            freeze=jsondecode(fileread(fullfile(protocol.storage,'final_freeze.json')));
            assert(freeze.all_models_and_calibrations_frozen);
        end
        cfg=original;
        cfg.types.enabled={'FMB','AMA','SawFM','SinFM','MFT','ISRJ'};
        cfg.types.known=cfg.types.enabled;
        cfg.types.unknown={};
        cfg.dataset.min_components=1;
        cfg.dataset.max_components=6;
        cfg.dataset.combination_mode='all';
        perPattern=96;
        if strcmp(role,'fit_pool'), perPattern=480; end
        if strcmp(role,'confirmation_pool'), perPattern=64; end
        cfg.dataset.samples_per_pattern=perPattern;
        cfg.dataset.num_samples=63*perPattern;
        cfg.seed=protocol.pool_seeds.(split).(role);
        cfg.output_dir=folder;
        cfg.output.save_v73=true;
        cfg.output.save_legacy=false;
        cfg=normalize_and_validate_config(cfg,'balanced');
        assert(isequal(cfg.params.common.JNR_dB(:)',[10,30]));
        assert(cfg.derived.sample_length==3000);
        registry=build_jamming_registry(cfg);
        schema=build_parameter_schema(registry);
        manifest=build_closedset_manifest(cfg,registry,'train');
        if ~isfolder(folder), mkdir(folder); end
        write_json(fullfile(folder,'generation_config.json'),cfg);
        write_json(fullfile(folder,'generation_manifest.json'),table2struct(manifest));
        result=run_closedset_splits(cfg,registry,schema,true,'train','error');
        write_json(receiptPath,struct('role',role,'path',result.train_path,'base_seed',cfg.seed,'samples',height(manifest),'status',result.splits.train.status));
        fprintf('POOL_COMPLETE %s %s\n',split,role);
    end
end
function value=row_ranges(value)
    names=fieldnames(value);
    for k=1:numel(names)
        field=names{k};
        if isstruct(value.(field))
            value.(field)=row_ranges(value.(field));
        elseif isnumeric(value.(field)) && isvector(value.(field))
            value.(field)=value.(field)(:)';
        end
    end
end
function write_json(path,value)
    fid=fopen(path,'w'); assert(fid~=-1);
    cleaner=onCleanup(@()fclose(fid));
    fwrite(fid,unicode2native(jsonencode(value,PrettyPrint=true),'UTF-8'),'uint8');
end
