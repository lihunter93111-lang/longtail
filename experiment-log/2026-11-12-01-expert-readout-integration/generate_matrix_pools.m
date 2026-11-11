function generate_matrix_pools(protocolPath,matrixPath)
    protocol=jsondecode(fileread(protocolPath));matrix=jsondecode(fileread(matrixPath));
    generatorRoot=fullfile(protocol.repository,'data_generation','0716','可配置组合干扰数据生成');
    addpath(generatorRoot,fullfile(generatorRoot,'core'),fullfile(generatorRoot,'generators'));
    original=jsondecode(fileread(protocol.generator_config));original.params=row_ranges(original.params);
    for j=1:numel(matrix.new_pools)
        pool=matrix.new_pools(j);folder=fullfile(protocol.storage,'matrix',pool.split,'data',pool.role);
        receipt=fullfile(folder,'generation_receipt.json');if isfile(receipt),continue;end
        cfg=original;cfg.types.enabled={'FMB','AMA','SawFM','SinFM','MFT','ISRJ'};cfg.types.known=cfg.types.enabled;cfg.types.unknown={};
        cfg.dataset.min_components=1;cfg.dataset.max_components=6;cfg.dataset.combination_mode='all';
        cfg.dataset.samples_per_pattern=pool.per_pattern;cfg.dataset.num_samples=63*pool.per_pattern;
        cfg.seed=pool.seed;cfg.output_dir=folder;cfg.output.save_v73=true;cfg.output.save_legacy=false;
        cfg=normalize_and_validate_config(cfg,'balanced');assert(cfg.derived.sample_length==3000);
        registry=build_jamming_registry(cfg);schema=build_parameter_schema(registry);manifest=build_closedset_manifest(cfg,registry,'train');
        if ~isfolder(folder),mkdir(folder);end
        write_json(fullfile(folder,'generation_config.json'),cfg);write_json(fullfile(folder,'generation_manifest.json'),table2struct(manifest));
        result=run_closedset_splits(cfg,registry,schema,true,'train','error');
        write_json(receipt,struct('role',pool.role,'path',result.train_path,'base_seed',pool.seed,'samples',height(manifest),'status',result.splits.train.status));
        fprintf('MATRIX_POOL_COMPLETE %s %s\n',pool.split,pool.role);
    end
end
function value=row_ranges(value)
    names=fieldnames(value);
    for k=1:numel(names)
        name=names{k};if isstruct(value.(name)),value.(name)=row_ranges(value.(name));
        elseif isnumeric(value.(name)) && isvector(value.(name)),value.(name)=value.(name)(:)';end
    end
end
function write_json(path,value)
    fid=fopen(path,'w');assert(fid~=-1);cleaner=onCleanup(@()fclose(fid));fwrite(fid,unicode2native(jsonencode(value,PrettyPrint=true),'UTF-8'),'uint8');
end
