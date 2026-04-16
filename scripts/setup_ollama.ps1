# Instala y prepara Ollama + modelos locales en Windows.
# Requiere permisos de administrador la primera vez.

Write-Host "1) Instalando Ollama (si no existe)..."
if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
    winget install --id Ollama.Ollama -e --accept-package-agreements --accept-source-agreements
} else {
    Write-Host "   Ollama ya instalado."
}

Write-Host "2) Arrancando servicio..."
Start-Process -FilePath "ollama" -ArgumentList "serve" -WindowStyle Hidden

Start-Sleep -Seconds 3

Write-Host "3) Descargando modelos (primera vez puede tardar)..."
ollama pull llama3
ollama pull nomic-embed-text

Write-Host "4) Modelos disponibles:"
ollama list
