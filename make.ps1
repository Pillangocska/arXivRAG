<#
.SYNOPSIS
    PowerShell task runner mirroring the project Makefile for Windows.

.DESCRIPTION
    Provides the same tasks as the Makefile without requiring GNU make.
    Run targets the same way you would with make:

        .\make.ps1 up                   # start Qdrant only
        .\make.ps1 up -Service all      # start both Qdrant and the app
        .\make.ps1 ingest
        .\make.ps1 run -Q "your question"
        .\make.ps1 test
        .\make.ps1 eval -EvalSize 10

.PARAMETER Target
    The task to run. See .\make.ps1 help for the full list.

.PARAMETER Q
    The question passed to the run / run-docker targets.

.PARAMETER EvalSize
    Number of samples for the eval-generate target. Defaults to 10.

.PARAMETER Service
    Service(s) to start with the up target: qdrant (default) or all (both Qdrant and app).
#>
[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [ValidateSet(
        'help', 'up', 'down', 'ingest', 'run', 'test', 'eval',
        'eval-generate', 'eval-run', 'build', 'ingest-docker',
        'run-docker', 'clean'
    )]
    [string]$Target = 'help',

    [string]$Q = '',

    [int]$EvalSize = 10,

    [ValidateSet('qdrant', 'app', 'all')]
    [string]$Service = 'qdrant'
)

$ErrorActionPreference = 'Stop'

function Invoke-Step {
    param([Parameter(Mandatory)][scriptblock]$Step)
    & $Step
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code $LASTEXITCODE."
    }
}

function Show-Help {
    Write-Host 'Targets:'
    Write-Host '  up [-Service qdrant|all]      Start services (default: qdrant only).'
    Write-Host '  down                          Stop all services.'
    Write-Host '  ingest                        Build the index on the host (uv).'
    Write-Host '  run -Q "..."                  Ask a question on the host (uv).'
    Write-Host '  test                          Run the test suite.'
    Write-Host '  eval                          Generate an eval set and score it with Ragas.'
    Write-Host '  build                         Build the app container image.'
    Write-Host '  ingest-docker                 Build the index inside the container.'
    Write-Host '  run-docker -Q "..."           Ask a question inside the container.'
    Write-Host '  clean                         Remove caches and eval results.'
}

switch ($Target) {
    'help' { Show-Help }

    # --- Services ---
    'up' {
        $services = switch ($Service) {
            'qdrant' { @('qdrant') }
            'app'    { @('qdrant', 'app') }
            'all'    { @('qdrant', 'app') }
        }
        Invoke-Step { docker compose up -d $services }
    }
    'down' { Invoke-Step { docker compose down } }

    # --- Host (uv) workflow ---
    'ingest' { Invoke-Step { uv run python -m arxiv_rag.app ingest } }

    'run' {
        if ([string]::IsNullOrWhiteSpace($Q)) {
            Write-Host 'Usage: .\make.ps1 run -Q "your question"'
            exit 1
        }
        Invoke-Step { uv run python -m arxiv_rag.app ask $Q }
    }

    'test' { Invoke-Step { uv run pytest -q } }

    'eval' {
        Invoke-Step { uv run python -m eval.generate --size $EvalSize }
        Invoke-Step { uv run python -m eval.run_eval }
    }

    'eval-generate' {
        Invoke-Step { uv run python -m eval.generate --size $EvalSize }
    }

    'eval-run' { Invoke-Step { uv run python -m eval.run_eval } }

    # --- Containerized workflow ---
    'build' { Invoke-Step { docker compose build app } }

    'ingest-docker' { Invoke-Step { docker compose run --rm --no-deps app ingest } }

    'run-docker' {
        if ([string]::IsNullOrWhiteSpace($Q)) {
            Write-Host 'Usage: .\make.ps1 run-docker -Q "your question"'
            exit 1
        }
        Invoke-Step { docker compose run --rm --no-deps app ask $Q }
    }

    # --- Cleanup ---
    'clean' {
        foreach ($path in '.pytest_cache', '.ruff_cache', 'eval/results') {
            if (Test-Path $path) {
                Remove-Item -Recurse -Force $path
            }
        }
    }
}
