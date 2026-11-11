function generate_frozen_pairs(repoRoot, experimentRoot, split)
% Reuse audited physical interventions after the campaign-wide model freeze.
    freeze = jsondecode(fileread(fullfile(experimentRoot,'final_freeze.json')));
    assert(freeze.all_models_and_calibrations_frozen);
    source = jsondecode(fileread(fullfile(experimentRoot,split,'confirmation','manifest.json')));
    outputRoot = fullfile(experimentRoot,split,'confirmation_pairs');
    if ~isfolder(outputRoot)
        mkdir(outputRoot);
    end
    assert(~isfile(fullfile(outputRoot,'evaluation_access.json')));
    roles = struct('dev_path',source.path,'development',source.indices, ...
        'source_role','frozen_confirmation_physical_diagnostic','freeze_sha256',source.freeze_sha256);
    write_json(fullfile(outputRoot,'manifest.json'),roles);
    addpath(fullfile(repoRoot,'experiment-log','2026-11-08-01-overnight'));
    addpath(fullfile(repoRoot,'experiment-log','2026-11-09-01-wide-followup'));
    generate_confirmation_pairs(repoRoot,outputRoot);
    generate_absent_views(repoRoot,outputRoot);
    receiptPath = fullfile(outputRoot,'absent_pairing_receipt.json');
    receipt = jsondecode(fileread(receiptPath));
    receipt.original_generator_role = receipt.role;
    receipt.role = 'frozen_confirmation_physical_diagnostic';
    receipt.freeze_sha256 = source.freeze_sha256;
    write_json(receiptPath,receipt);
    fprintf('FROZEN_PHYSICAL_PAIRS_VERIFIED %s\n',split);
end

function write_json(path,value)
    fid = fopen(path,'w');
    assert(fid~=-1);
    cleaner = onCleanup(@() fclose(fid));
    fwrite(fid,unicode2native(jsonencode(value,PrettyPrint=true),'UTF-8'),'uint8');
end
