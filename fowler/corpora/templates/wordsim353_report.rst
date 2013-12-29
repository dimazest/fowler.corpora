Wordsim 353
===========

==================== ============== ===========
            Measure   Spearman rho     p-value
==================== ============== ===========
{%- for m, (rho, p) in cor_coof %}
{%- set m = m.rjust(19) %}
{%- set rho = '{:0.3f}'.format(rho).rjust(13) %}
{%- set p = '{:0.3e}'.format(p).rjust(12) %}
{{ m              }} {{ rho      }} {{ p     }}
{%- endfor %}
==================== ============== ===========
