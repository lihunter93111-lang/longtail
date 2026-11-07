function generate_lt_abc_candidates(repoRoot, outputRoot, sourcePath)
% Fresh candidates using the original full-power physical configuration.
    generatorRoot = fullfile(repoRoot,'data_generation','0716','可配置组合干扰数据生成');
    addpath(generatorRoot,fullfile(generatorRoot,'core'),fullfile(generatorRoot,'generators'));
    original = load(sourcePath,'config');
    labels = {'FMB','AMA','SawFM','SinFM','MFT','ISRJ'};
    for role = {'fit_pool','dev_pool'}
        name=role{1}; cfg=original.config;
        cfg.types.enabled=labels; cfg.types.known=labels; cfg.types.unknown={};
        cfg.dataset.min_components=1; cfg.dataset.max_components=6;
        cfg.dataset.combination_mode='all';
        if strcmp(name,'fit_pool'); cfg.seed=3106110600; n=400;
        else; cfg.seed=3106210600; n=96; end
        cfg.dataset.samples_per_pattern=n; cfg.dataset.num_samples=63*n;
        cfg.output_dir=fullfile(outputRoot,name);
        cfg.output.save_v73=true; cfg.output.save_legacy=false;
        cfg=normalize_and_validate_config(cfg,'balanced');
        assert(isequal(cfg.params.common.JNR_dB,[10,30]));
        assert(cfg.derived.sample_length==3000);
        registry=build_jamming_registry(cfg); schema=build_parameter_schema(registry);
        manifest=build_closedset_manifest(cfg,registry,'train');
        if ~isfolder(cfg.output_dir); mkdir(cfg.output_dir); end
        % Keep sample_seed and active component order directly readable.
        rows=table2struct(manifest);
        fid=fopen(fullfile(cfg.output_dir,'generation_manifest.json'),'w');
        fwrite(fid,unicode2native(jsonencode(rows),'UTF-8'),'uint8'); fclose(fid);
        fid=fopen(fullfile(cfg.output_dir,'generation_config.json'),'w');
        fwrite(fid,unicode2native(jsonencode(cfg),'UTF-8'),'uint8'); fclose(fid);
        result=run_closedset_splits(cfg,registry,schema,true,'train','error');
        receipt=struct('role',name,'base_seed',cfg.seed,'source_config',sourcePath, ...
            'path',result.train_path,'samples',height(manifest),'status',result.splits.train.status, ...
            'locked_test_used',false,'physical_config','copied_from_original_full_power_10_30dB_v1');
        fid=fopen(fullfile(cfg.output_dir,'generation_receipt.json'),'w');
        fwrite(fid,unicode2native(jsonencode(receipt,PrettyPrint=true),'UTF-8'),'uint8'); fclose(fid);
        fprintf('LT_ABC_CANDIDATE_COMPLETE %s %s\n',name,result.train_path);
    end
end
