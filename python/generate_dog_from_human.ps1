$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
    python src_ortho_new\eval_vq_vae.py `
        .\models\model_dog `
        .\data\dog\train\D1_055_KAN01_001.bvh `
        generator `
        --input_path2 .\data\human\train\S01_A01.bvh `
        --training-stage prior `
        --decoder-source predicted_codebook `
        --prior-stage-architecture `
        --inference-temperature 0.5 `
        --no-stationary-root-lock

    if ($LASTEXITCODE -ne 0) { throw "Inference failed with exit code $LASTEXITCODE" }
    New-Item -ItemType Directory -Force -Path .\results | Out-Null
    Move-Item -LiteralPath .\data\eval_D1_055_KAN01_001.bvh -Destination .\results\dog_from_human.bvh -Force
    Write-Host 'Saved: results\dog_from_human.bvh'
}
finally {
    Pop-Location
}