(TeX-add-style-hook "main"
 (lambda ()
    (LaTeX-add-bibliographies
     "lsstdesc")
    (LaTeX-add-labels
     "sec:intro"
     "sec:design_notes"
     "fig:jla_X1_C"
     "sec:methods"
     "sec:results"
     "sec:discussion"
     "sec:conclusions"
     "tab:lse40"
     "tab:smtn002")
    (TeX-run-style-hooks
     "subfigure"
     "graphicx"
     "lsstdesc_macros"
     "docswitch"
     "acknowledgments"
     "contributions")))

