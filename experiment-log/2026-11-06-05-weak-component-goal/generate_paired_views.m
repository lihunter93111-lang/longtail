function generate_paired_views(repoRoot, dataRoot)
% Same-identity interventions, verified before per-input RMS normalization.
    generatorRoot = fullfile(repoRoot,'data_generation','0716','可配置组合干扰数据生成');
    addpath(generatorRoot,fullfile(generatorRoot,'core'),fullfile(generatorRoot,'generators'));
    roles = jsondecode(fileread(fullfile(dataRoot,'manifest.json')));
    labels = {'FMB','AMA','SawFM','SinFM','MFT','ISRJ'};
    targets = [3,5,6];
    for mode = {'fit','diagnostic'}
        modeName = mode{1};
        receiptPath = fullfile(dataRoot,[modeName,'_pairing_receipt.json']);
        if isfile(receiptPath)
            fprintf('REUSE_VERIFIED_VIEWS %s\n',modeName);
            continue
        end
        isFit = strcmp(modeName,'fit');
        if isFit
            source = roles.fit_path;
            indices = roles.fit_A;
            gaps = 10;
        else
            source = roles.dev_path;
            indices = roles.development;
            gaps = [0,5,10,15];
        end
        original = load(source,'config');
        cfg = original.config;
        registry = build_jamming_registry(cfg);
        schema = build_parameter_schema(registry);
        manifest = build_closedset_manifest(cfg,registry,'train');
        reference = build_reference_signal(cfg);
        sourceData = load(source,'iq_complex');
        n = numel(indices);
        records = cell(n,1);
        arrays = cell(numel(gaps),1);
        for g = 1:numel(gaps)
            arrays{g} = struct('x_real',zeros(n,3000,'single'), ...
                'x_imag',zeros(n,3000,'single'),'positions',zeros(n,1), ...
                'target_labels',zeros(n,1),'effective_JNR_dB',nan(n,6), ...
                'parameter_values',{cell(n,1)},'parameter_mask',{cell(n,1)},'count',0);
        end
        maxRebuild = 0;
        maxDelta = 0;
        maxRelativeDelta = 0;
        for position = 1:n
            rawIndex = indices(position)+1;
            row = manifest(rawIndex,:);
            sample = compose_jamming_sample(cfg,registry,schema,row,reference);
            saved = sourceData.iq_complex(rawIndex,:);
            rebuildError = max(abs(sample.iq_complex-saved));
            assert(rebuildError <= 1e-9*max(1,max(abs(saved))));
            maxRebuild = max(maxRebuild,rebuildError);
            power = sample.component_effective_JNR_dB;
            active = sample.active_type_codes;
            candidates = [];
            maxima = nan(1,6);
            for label = targets
                component = find(strcmp(active,labels{label}));
                if isempty(component) || numel(power)<2
                    continue
                end
                other = power;
                other(component) = -inf;
                maxima(label) = max(other);
                eligible = maxima(label)>=20;
                if isFit
                    eligible = eligible && power(component)>maxima(label)-10;
                end
                if eligible
                    candidates(end+1) = label;
                end
            end
            record = struct('position',position-1,'raw_row',rawIndex-1, ...
                'sample_seed',sample.sample_seed,'receiver_noise_seed',sample.receiver_noise_seed, ...
                'component_seeds',sample.component_seeds,'eligible',~isempty(candidates), ...
                'target_label',-1,'original_JNR_dB',NaN,'max_other_JNR_dB',NaN, ...
                'gap15_feasible',false,'reason','no_eligible_target', ...
                'original_reconstruction_max_abs',rebuildError);
            if ~isempty(candidates)
                label = candidates(mod(sample.sample_seed,numel(candidates))+1);
                component = find(strcmp(active,labels{label}));
                record.target_label = label-1;
                record.original_JNR_dB = power(component);
                record.max_other_JNR_dB = maxima(label);
                record.gap15_feasible = maxima(label)>=25;
                record.reason = 'eligible';
                targetStream = RandStream('mt19937ar','Seed',sample.component_seeds(component));
                [clean,params] = generate_single_jamming(labels{label},cfg,registry,targetStream,reference);
                assert(abs(params.JNR_dB-power(component))<1e-10);
                originalTarget = calibrate_component_power(clean,10*log10(cfg.radar.noise_power)+power(component));
                for g = 1:numel(gaps)
                    gap = gaps(g);
                    if gap==15 && ~record.gap15_feasible
                        continue
                    end
                    newPower = maxima(label)-gap;
                    assert(newPower>=10 && newPower<=30);
                    changedRow = row;
                    offsets = zeros(1,numel(power));
                    offsets(component) = newPower-power(component);
                    changedRow.component_target_power_dB{1} = offsets;
                    weakMask = false(1,numel(power));
                    weakMask(component) = gap>=10;
                    changedRow.weak_component_mask{1} = weakMask;
                    changed = compose_jamming_sample(cfg,registry,schema,changedRow,reference);
                    assert(isequal(changed.component_seeds,sample.component_seeds));
                    assert(changed.receiver_noise_seed==sample.receiver_noise_seed);
                    assert(isequal(changed.known_label,sample.known_label));
                    expectedPower = power;
                    expectedPower(component) = newPower;
                    assert(max(abs(changed.component_effective_JNR_dB-expectedPower))<1e-10);
                    delta = originalTarget*(10^((newPower-power(component))/20)-1);
                    residual = max(abs(changed.iq_complex-sample.iq_complex-delta));
                    relativeResidual = residual/max(1,max(abs(changed.iq_complex)));
                    assert(relativeResidual<1e-12);
                    maxDelta = max(maxDelta,residual);
                    maxRelativeDelta = max(maxRelativeDelta,relativeResidual);
                    k = arrays{g}.count+1;
                    arrays{g}.x_real(k,:) = single(real(changed.iq_complex));
                    arrays{g}.x_imag(k,:) = single(imag(changed.iq_complex));
                    arrays{g}.positions(k) = position-1;
                    arrays{g}.target_labels(k) = label-1;
                    arrays{g}.effective_JNR_dB(k,logical(sample.known_label)) = expectedPower;
                    arrays{g}.parameter_values{k} = changed.parameter_values;
                    arrays{g}.parameter_mask{k} = changed.parameter_mask;
                    arrays{g}.count = k;
                end
            elseif numel(power)<2
                record.reason = 'single_component';
            elseif ~any(ismember(active,labels(targets)))
                record.reason = 'target_absent';
            elseif ~any(maxima(targets)>=20)
                record.reason = 'other_max_below_20';
            else
                record.reason = 'all_feasible_targets_already_gap_ge10';
            end
            records{position} = record;
            if mod(position,500)==0
                fprintf('PAIR_PROGRESS %s %d/%d\n',modeName,position,n);
            end
        end
        counts = zeros(size(gaps));
        for g = 1:numel(gaps)
            output = arrays{g};
            k = output.count;
            counts(g) = k;
            fields = fieldnames(output);
            for f = 1:numel(fields)
                field = fields{f};
                if ~strcmp(field,'count')
                    output.(field) = output.(field)(1:k,:);
                end
            end
            output.gap_dB = gaps(g);
            path = fullfile(dataRoot,sprintf('%s_gap%d.mat',modeName,gaps(g)));
            save(path,'-struct','output','-v7.3');
        end
        write_json(fullfile(dataRoot,[modeName,'_pairing_identities.json']),vertcat(records{:}));
        receipt = struct('status','verified','mode',modeName,'identity_count',n, ...
            'gaps',gaps,'view_counts',counts,'raw_reconstruction_max_abs',maxRebuild, ...
            'intervention_delta_max_abs',maxDelta,'intervention_delta_max_relative',maxRelativeDelta, ...
            'single_target_delta_verified',true,'locked_test_used',false);
        write_json(receiptPath,receipt);
        fprintf('PAIRING_VERIFIED %s %s\n',modeName,jsonencode(receipt));
    end
end

function write_json(path,value)
    fid = fopen(path,'w');
    assert(fid~=-1);
    cleaner = onCleanup(@() fclose(fid));
    fwrite(fid,unicode2native(jsonencode(value,PrettyPrint=true),'UTF-8'),'uint8');
end
