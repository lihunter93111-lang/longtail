function generate_fresh_candidates(repoRoot, outputRoot, sourcePath, promptPath)
% Fresh identities authorized by the new, separately frozen research protocol.
    assert(isfile(promptPath));
    generatorRoot = fullfile(repoRoot,'data_generation','0716','可配置组合干扰数据生成');
    addpath(generatorRoot,fullfile(generatorRoot,'core'),fullfile(generatorRoot,'generators'));
    original = load(sourcePath,'config');
    labels = {'FMB','AMA','SawFM','SinFM','MFT','ISRJ'};
    for role = {'fit_pool','dev_pool'}
        name = role{1};
        if isfile(fullfile(outputRoot,name,'generation_receipt.json'))
            fprintf('REUSE_COMPLETED_POOL %s\n',name);
            continue
        end
        cfg = original.config;
        cfg.types.enabled = labels;
        cfg.types.known = labels;
        cfg.types.unknown = {};
        cfg.dataset.min_components = 1;
        cfg.dataset.max_components = 6;
        cfg.dataset.combination_mode = 'all';
        if strcmp(name,'fit_pool')
            cfg.seed = 3107110600;
            perPattern = 400;
        else
            cfg.seed = 3107210600;
            perPattern = 96;
        end
        cfg.dataset.samples_per_pattern = perPattern;
        cfg.dataset.num_samples = 63 * perPattern;
        cfg.output_dir = fullfile(outputRoot,name);
        cfg.output.save_v73 = true;
        cfg.output.save_legacy = false;
        cfg = normalize_and_validate_config(cfg,'balanced');
        assert(isequal(cfg.params.common.JNR_dB,[10,30]));
        assert(cfg.derived.sample_length==3000);
        registry = build_jamming_registry(cfg);
        schema = build_parameter_schema(registry);
        manifest = build_closedset_manifest(cfg,registry,'train');
        if ~isfolder(cfg.output_dir)
            mkdir(cfg.output_dir);
        end
        fid = fopen(fullfile(cfg.output_dir,'generation_manifest.json'),'w');
        fwrite(fid,unicode2native(jsonencode(table2struct(manifest)),'UTF-8'),'uint8');
        fclose(fid);
        fid = fopen(fullfile(cfg.output_dir,'generation_config.json'),'w');
        fwrite(fid,unicode2native(jsonencode(cfg),'UTF-8'),'uint8');
        fclose(fid);
        result = run_closedset_splits(cfg,registry,schema,true,'train','error');
        receipt = struct('role',name,'base_seed',cfg.seed,'source_config',sourcePath, ...
            'path',result.train_path,'samples',height(manifest), ...
            'status',result.splits.train.status,'goal_prompt_path',promptPath, ...
            'locked_test_used',false);
        fid = fopen(fullfile(cfg.output_dir,'generation_receipt.json'),'w');
        fwrite(fid,unicode2native(jsonencode(receipt,PrettyPrint=true),'UTF-8'),'uint8');
        fclose(fid);
        fprintf('BCE_CONFIRMATION_GENERATED %s %s\n',name,result.train_path);
    end
end
