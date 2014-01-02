Hyper parameter estimation
==========================

:paper: {{ paper }}
:accuracy: {{ accuracy.round(3) }}
{%- for key, value in store_metadata.items()  %}
:{{  key }}: {{ value }}
{%- endfor %}
:command: {{ argv }}

Grid accuracy scores on development set::

    {% for s in clf.grid_scores_ %}
    {{ s.mean_validation_score|round(3) }} (+/-{{ (s.cv_validation_scores.std() / 2.0)|round(3) }}) for {{s.parameters}}
    {%- endfor %}

Evaluation results
------------------

==================== ========== ========== ========== ==========
                tag  precision     recall   f1-score    support
==================== ========== ========== ========== ==========
{%- for t, p, r, f, s in tprfs %}
{%- set t = t.replace('+', '\+').replace('_', '\_').rjust(19) %}
{%- set p = '{:0.3f}'.format(p).rjust(10) %}
{%- set r = '{:0.3f}'.format(r).rjust(10) %}
{%- set f = '{:0.3f}'.format(f).rjust(10) %}
{%- set s = (s|string).rjust(10) %}
{{ t              }} {{ p    }} {{ r    }} {{ f    }} {{ s    }}
{%- endfor %}
-------------------- ---------- ---------- ---------- ----------
{%- set p_avg = '{:0.3f}'.format(p_avg).rjust(10) %}
{%- set r_avg = '{:0.3f}'.format(r_avg).rjust(10) %}
{%- set f_avg = '{:0.3f}'.format(f_avg).rjust(10) %}
{%- set s_sum = (s_sum|string).rjust(10) %}
  weighted avg/total {{ p_avg}} {{ r_avg}} {{ f_avg}} {{ s_sum}}
==================== ========== ========== ========== ==========

The model is trained on the full development set.
The scores are computed on the full evaluation set.
