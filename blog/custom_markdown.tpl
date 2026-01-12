{% extends 'markdown/index.md.j2' %}

{% block stream %}
```text
{{ output.text | strip_ansi }}
```
{% endblock stream %}

{% block execute_result %}
```text
{{ output.data['text/plain'] | strip_ansi }}
```
{% endblock execute_result %}