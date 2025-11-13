{{/*
Expand the name of the chart.
*/}}
{{- define "mae.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
*/}}
{{- define "mae.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "mae.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "mae.labels" -}}
helm.sh/chart: {{ include "mae.chart" . }}
{{ include "mae.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "mae.selectorLabels" -}}
app.kubernetes.io/name: {{ include "mae.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "mae.serviceAccountName" -}}
{{- if .Values.security.serviceAccount.create }}
{{- default (include "mae.fullname" .) .Values.security.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.security.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Return the proper image name for API
*/}}
{{- define "mae.api.image" -}}
{{- printf "%s:%s" .Values.api.image.repository .Values.api.image.tag }}
{{- end }}

{{/*
Return the proper image name for Redis
*/}}
{{- define "mae.redis.image" -}}
{{- printf "%s:%s" .Values.redis.image.repository .Values.redis.image.tag }}
{{- end }}

{{/*
Return the proper image name for ChromaDB
*/}}
{{- define "mae.chromadb.image" -}}
{{- printf "%s:%s" .Values.chromadb.image.repository .Values.chromadb.image.tag }}
{{- end }}
