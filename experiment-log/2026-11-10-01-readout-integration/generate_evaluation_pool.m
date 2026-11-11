function generate_evaluation_pool(repoRoot,outputRoot,sourcePath,freezePath,confirmationSeed,role)
% Generate the reserved final cohort after the candidate freeze exists.
    if strcmp(role,'confirmation')
        freeze = jsondecode(fileread(freezePath));
        assert(freeze.all_models_and_calibrations_frozen);
    end
    generatorRoot = fullfile(repoRoot,'data_generation','0716','可配置组合干扰数据生成');
    addpath(generatorRoot,fullfile(generatorRoot,'core'),fullfile(generatorRoot,'generators'));
    poolRoot = fullfile(outputRoot,'evaluation_pool');
    assert(~isfile(fullfile(outputRoot,'evaluation_access.json')));
    if isfile(fullfile(poolRoot,'generation_receipt.json'))
        fprintf('FINAL_POOL_ALREADY_GENERATED\n');
        return
    end
    original = load(sourcePath,'config');
    cfg = original.config;
    labels = {'FMB','AMA','SawFM','SinFM','MFT','ISRJ'};
    cfg.types.enabled = labels;
    cfg.types.known = labels;
    cfg.types.unknown = {};
    cfg.dataset.min_components = 1;
    cfg.dataset.max_components = 6;
    cfg.dataset.combination_mode = 'all';
    cfg.dataset.samples_per_pattern = 64;
    cfg.dataset.num_samples = 4032;
    cfg.seed = confirmationSeed;
    cfg.output_dir = poolRoot;
    cfg.output.save_v73 = true;
    cfg.output.save_legacy = false;
    cfg = normalize_and_validate_config(cfg,'balanced');
    registry = build_jamming_registry(cfg);
    schema = build_parameter_schema(registry);
    manifest = build_closedset_manifest(cfg,registry,'train');
    if ~isfolder(poolRoot)
        mkdir(poolRoot);
    end
    write_json(fullfile(poolRoot,'generation_manifest.json'),table2struct(manifest));
    write_json(fullfile(poolRoot,'generation_config.json'),cfg);
    result = run_closedset_splits(cfg,registry,schema,true,'train','error');
    receipt = struct('role',role,'base_seed',cfg.seed, ...
        'path',result.train_path,'samples',height(manifest),'status',result.splits.train.status, ...
        'freeze_path',freezePath,'source_config',sourcePath,'locked_test_used',false);
    write_json(fullfile(poolRoot,'generation_receipt.json'),receipt);
    fprintf('FINAL_CONFIRMATION_GENERATED %s\n',result.train_path);
end

function write_json(path,value)
    fid = fopen(path,'w');
    assert(fid~=-1);
    cleaner = onCleanup(@() fclose(fid));
    fwrite(fid,unicode2native(jsonencode(value,PrettyPrint=true),'UTF-8'),'uint8');
end
