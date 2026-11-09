function generate_absent_views(repoRoot,dataRoot)
% Remove the paired target while preserving all other physical components.
    generatorRoot = fullfile(repoRoot,'data_generation','0716','可配置组合干扰数据生成');
    addpath(generatorRoot,fullfile(generatorRoot,'core'),fullfile(generatorRoot,'generators'));
    roles = jsondecode(fileread(fullfile(dataRoot,'manifest.json')));
    records = jsondecode(fileread(fullfile(dataRoot,'diagnostic_pairing_identities.json')));
    cfgData = load(roles.dev_path,'config');
    cfg = cfgData.config;
    registry = build_jamming_registry(cfg);
    schema = build_parameter_schema(registry);
    manifest = build_closedset_manifest(cfg,registry,'train');
    reference = build_reference_signal(cfg);
    raw = load(roles.dev_path,'iq_complex');
    labels = {'FMB','AMA','SawFM','SinFM','MFT','ISRJ'};
    selected = records([records.eligible]);
    n = numel(selected);
    x_real = zeros(n,3000,'single');
    x_imag = zeros(n,3000,'single');
    positions = zeros(n,1);
    target_labels = zeros(n,1);
    known_label = false(n,6);
    effective_JNR_dB = nan(n,6);
    parameter_values = cell(n,1);
    parameter_mask = cell(n,1);
    maximumResidual = 0;
    maximumRelativeResidual = 0;
    for k = 1:n
        record = selected(k);
        row = manifest(record.raw_row+1,:);
        original = compose_jamming_sample(cfg,registry,schema,row,reference);
        assert(max(abs(original.iq_complex-raw.iq_complex(record.raw_row+1,:)))==0);
        label = record.target_label+1;
        component = find(strcmp(original.active_type_codes,labels{label}));
        assert(numel(component)==1 && original.combination_size>=2);
        stream = RandStream('mt19937ar','Seed',original.component_seeds(component));
        [signal,params] = generate_single_jamming(labels{label},cfg,registry,stream,reference);
        signal = calibrate_component_power(signal,10*log10(cfg.radar.noise_power)+params.JNR_dB);
        ids = row.active_type_ids{1};
        codes = row.active_type_codes{1};
        ids(component) = [];
        codes(component) = [];
        row.active_type_ids{1} = ids;
        row.active_type_codes{1} = codes;
        row.combination_id{1} = strjoin(codes,'+');
        row.combination_size = numel(ids);
        row.component_target_power_dB{1} = zeros(1,numel(ids));
        row.weak_component_mask{1} = false(1,numel(ids));
        absent = compose_jamming_sample(cfg,registry,schema,row,reference);
        expectedSeeds = original.component_seeds;
        expectedSeeds(component) = [];
        expectedLabels = original.known_label;
        expectedLabels(label) = false;
        assert(isequal(absent.component_seeds,expectedSeeds));
        assert(absent.receiver_noise_seed==original.receiver_noise_seed);
        assert(isequal(absent.known_label,expectedLabels));
        residual = max(abs(absent.iq_complex-original.iq_complex+signal));
        relative = residual/max(1,max(abs(original.iq_complex)));
        assert(relative<1e-12);
        maximumResidual = max(maximumResidual,residual);
        maximumRelativeResidual = max(maximumRelativeResidual,relative);
        x_real(k,:) = single(real(absent.iq_complex));
        x_imag(k,:) = single(imag(absent.iq_complex));
        positions(k) = record.position;
        target_labels(k) = record.target_label;
        known_label(k,:) = absent.known_label;
        effective_JNR_dB(k,absent.known_label) = absent.component_effective_JNR_dB;
        parameter_values{k} = absent.parameter_values;
        parameter_mask{k} = absent.parameter_mask;
        if mod(k,500)==0
            fprintf('ABSENT_PROGRESS %d/%d\n',k,n);
        end
    end
    save(fullfile(dataRoot,'diagnostic_absent.mat'),'x_real','x_imag','positions','target_labels', ...
        'known_label','effective_JNR_dB','parameter_values','parameter_mask','-v7.3');
    receipt = struct('status','verified','identities',n,'max_abs_residual',maximumResidual, ...
        'max_relative_residual',maximumRelativeResidual,'receiver_noise_and_other_components_verified',true, ...
        'label_and_parameter_metadata_updated',true,'role','adaptive_development_counterfactual');
    fid = fopen(fullfile(dataRoot,'absent_pairing_receipt.json'),'w');
    fwrite(fid,unicode2native(jsonencode(receipt,PrettyPrint=true),'UTF-8'),'uint8');
    fclose(fid);
    fprintf('ABSENT_VIEWS_VERIFIED %s\n',jsonencode(receipt));
end
