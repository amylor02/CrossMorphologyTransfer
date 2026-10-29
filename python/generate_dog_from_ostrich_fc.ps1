$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
    python src_ortho_new\eval_vq_vae.py `
        .\models\model_dog_contact `
        .\data\dog\train\D1_055_KAN01_001.bvh `
        generator `
        --input_path2 .\data\ostrich\train\idle_walk_idle_2.bvh `
        --training-stage prior `
        --decoder-source predicted_codebook `
        --prior-stage-architecture `
        --foot-contact-conditioning `
        --inference-temperature 0.5 `
        --no-stationary-root-lock

    if ($LASTEXITCODE -ne 0) { throw "Inference failed with exit code $LASTEXITCODE" }
    New-Item -ItemType Directory -Force -Path .\results | Out-Null
    Move-Item -LiteralPath .\data\eval_D1_055_KAN01_001.bvh -Destination .\results\dog_from_ostrich_fc.bvh -Force
    Write-Host 'Saved: results\dog_from_ostrich_fc.bvh'
}
finally {
    Pop-Location
}