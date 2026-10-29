$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
    python src_ortho_new\eval_vq_vae.py `
        .\models\model_human `
        .\data\human\train\S01_A01.bvh `
        generator `
        --input_path2 .\data\dog\train\D1_053_KAN01_001.bvh `
        --training-stage prior `
        --decoder-source predicted_codebook `
        --prior-stage-architecture `
        --inference-temperature 0.5 `
        --no-stationary-root-lock

    if ($LASTEXITCODE -ne 0) { throw "Inference failed with exit code $LASTEXITCODE" }
    New-Item -ItemType Directory -Force -Path .\results | Out-Null
    Move-Item -LiteralPath .\data\eval_S01_A01.bvh -Destination .\results\human_from_dog.bvh -Force
    Write-Host 'Saved: results\human_from_dog.bvh'
}
finally {
    Pop-Location
}