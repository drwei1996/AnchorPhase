@echo off
REM One-click reproduction of both new biological supplementary figures (SUPP DISCOVERY 01 + 02).
REM Requires: python (3.11+) with numpy, pandas, scipy, matplotlib, anndata, Pillow.
REM Inputs are frozen project files; nothing upstream is modified. Outputs overwrite only this folder.

set PY=py -X utf8
set CODE=%~dp0CODE

%PY% "%CODE%\supp_discovery_01_genome_wide_fit.py" || goto :err
%PY% "%CODE%\supp_discovery_01_analysis.py" || goto :err
%PY% "%CODE%\supp_discovery_01_ora.py" || goto :err
%PY% "%CODE%\supp_discovery_01_figure.py" || goto :err
%PY% "%CODE%\supp_discovery_02_figure.py" || goto :err

echo ALL SUPPLEMENTARY DISCOVERY FIGURES REBUILT SUCCESSFULLY
exit /b 0
:err
echo FAILED - see message above
exit /b 1
