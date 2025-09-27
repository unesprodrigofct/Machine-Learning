{{- define "ml-api.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "ml-api.fullname" -}}
{{- if .Values.fullnameOverride -}}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- $name := default .Chart.Name .Values.nameOverride -}}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- end -}}

{{- define "ml-api.labels" -}}
helm.sh/chart: {{ include "ml-api.chart" . }}
app.kubernetes.io/name: {{ include "ml-api.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/version: {{ .Chart.AppVersion }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end -}}

{{- define "ml-api.selectorLabels" -}}
app.kubernetes.io/name: {{ include "ml-api.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end -}}

{{- define "ml-api.chart" -}}
{{ .Chart.Name }}-{{ .Chart.Version }}
{{- end -}}

{{- define "ml-api.serviceAccountName" -}}
{{- if .Values.serviceAccount.create -}}
{{- default (include "ml-api.fullname" .) .Values.serviceAccount.name -}}
{{- else -}}
{{- default "default" .Values.serviceAccount.name -}}
{{- end -}}
{{- end -}}
