<h1>INDRA</h1>

Update 01 July 2020: This repository provides the code to generate your own variations on a seed file. For commercial purchase of projected files for simulation, based on climate model outputs, please email info@arbnco.com to request a quote.

This repository contains scripts to create synthetic weather time series from a short weather record of at least one year. This tool is named `indra` <sup>(1)</sup>. _All the scripts here should be treated as **experimental** unless explicitly stated otherwise._

<h2>Great, where do I begin?</h2>

<strong>Please, please see the <a href='https://github.com/paragrastogi/SyntheticWeather/wiki'>wiki</a> first.</strong> I can't make you do it though, so yes, be your own boss. The wiki contains a step-by-step guide to installing and running mighty <b>indra</b>.

If you know your way around MATLAB or Python, go directly into either the folder `m-files` (MATLAB files) or the folder `py-files` (Python files). Most of the scripts explain themselves. **Sample Python commands** are <a href='https://github.com/paragrastogi/SyntheticWeather/wiki/Sample-Commands'>given here</a>.

If you're interested in reading the methods used first, see the list of papers given below.

<h2>The methods</h2>

The MATLAB/R scripts are based on the algorithms published in Parag's thesis. While these scripts are well documented (in the two conference papers and thesis mentioned below), I won't be working on these any more. The Python scripts in the repository are translations of these original scripts. The method is an almost-completely-faithful translation of the MATLAB scripts<sup>(2)</sup>.

This work derives from my PhD thesis at the Ecole Polytechnique Federale de Lausanne, EPFL (**Chapter 3**). The method is described in the following publications/references:

1. P. Rastogi and M. Andersen, ‘Incorporating Climate Change Predictions in the Analysis of Weather-Based Uncertainty’, presented at the ASHRAE and IBPSA-USA Building Performance Modeling Conference, Salt Lake City, UT, USA, Aug. 2016, [Online]. Available: http://infoscience.epfl.ch/record/208743. 

>@inproceedings{rastogi2016incorporating,
>  title={Incorporating Climate Change Predictions in the Analysis of Weather-Based Uncertainty},
>  author={Rastogi, Parag and Andersen, Marilyne},
>  booktitle={Sixth National Conference of IBPSA-USA, Salt Lake City, UT},
>  year={2016}
>}

2. P. Rastogi, ‘On the sensitivity of buildings to climate: the interaction of weather and building envelopes in determining future building energy consumption’, PhD, Ecole polytechnique fédérale de Lausanne, Lausanne, Switzerland, 2016. Available: https://infoscience.epfl.ch/record/220971?ln=en. DOI: http://dx.doi.org/10.5075/epfl-thesis-6881.

>@phdthesis{rastogi2016sensitivity,
>  title={On the sensitivity of buildings to climate: the interaction of weather and building envelopes in determining future building energy consumption},
>  author={Rastogi, Parag},
>  year={2016},
>  school={Ecole Polytechnique F{\'e}d{\'e}rale de Lausanne}
>}

3. P. Rastogi and M. Andersen, ‘Embedding Stochasticity in Building Simulation Through Synthetic Weather Files’, presented at the 14th International Conference of the International Building Performance Simulation Association, Hyderabad, India, Dec. 2015, [Online]. Available: http://infoscience.epfl.ch/record/208743.

>@article{rastogi2015embedding,
>  title={Embedding stochasticity in building simulation through synthetic weather files},
>  author={Rastogi, Parag and Andersen, Marilyne},
>  journal={Proceedings of BS},
>  year={2015}
>}


<h2>License, implementation, and compatibility</h2>

This tool is distributed under the GPLv3 license. Please read what this means <a href='https://en.wikipedia.org/wiki/GNU_General_Public_License'>here</a>.

Using the older scripts requires only a valid MATLAB license and R (R is free to download and reuse). While you are free to use the scripts as you please, I am not liable for anything that happens as a result of using my scripts. Like if you accidentally release nuclear missiles, ruin the ski season in Switzerland, or cause a drought in Scotland.

<h2>Citation</h2>

Please cite this work using the citation for my thesis:

Rastogi, Parag. 2016. ‘On the Sensitivity of Buildings to Climate: The Interaction of Weather and Building Envelopes in Determining Future Building Energy Consumption’. PhD, Lausanne, Switzerland: Ecole polytechnique fédérale de Lausanne. EPFL Infoscience. https://infoscience.epfl.ch/record/220971?ln=en.

>@phdthesis{rastogi_sensitivity_2016,
>	address = {Lausanne, Switzerland},
>	type = {{PhD}},
>	title = {On the sensitivity of buildings to climate: the interaction of weather and building envelopes in determining future building energy consumption},
>	shorttitle = {Sensitivity of {Buildings} to {Climate}},
>	url = {https://infoscience.epfl.ch/record/220971?ln=en},
>	language = {EN},
>	school = {Ecole polytechnique fédérale de Lausanne},
>	author = {Rastogi, Parag},
>	month = aug,
>	year = {2016},
>	note = {doi:10.5075/epfl-thesis-6881}
>}

<h2>I'm panicking/clueless</h2>

If you have questions or concerns, or notice errors, please contact me at `contact[at]paragrastogi.com`.

Happy creating fake weather!


<h2>Footnotes</h2>

1. I haven't thought up a smarmy, contrived acronym yet but I'm working on it.

2. The difference lies in the 'simulation' of the SARMA model to produce synthetic 'de-mean-ed' series. I am not convinced that I should reproduce the 'custom noise' functionality used in the old scripts to simulate the SARMA models with bootstrapped residuals. For now, I am doing a 'conventional' simulation by using white noise.

<h2>Acknowledgements</h2>

1. This work began during my PhD at the Ecole Polytechnique Federale de Lausanne. It was funded by the CCEM SECURE project and the EuroTech consortium. February 2012 - August 2016
2.Subsequent work and the Python translation was carried out when I was a visiting scientist at the Energy Systems Research Unit (ESRU), University of Strathclyde, Glasgow, and the RIKEN Institute for Advanced Intelligence Project (RIKEN-AIP), Tokyo. My stay at these institutions was financed by the Swiss National Science Foundation (SNSF) under grant number p2elp2_168519.

I woud like to thank my hosts: Prof. Joe Clarke (Strathclyde, Glasgow) and Dr Mohammad Emtiyaz Khan and Prof. Masashi Sugiyama (RIKEN-AIP, Tokyo). The advice of Prof. Anthony Davison (EPFL, Lausanne) was crucial in creating the first models.

<h2>Disclaimer</h2>

Parag is the sole author of these scripts. The original PhD thesis was supervised by Professor Marilyne Andersen. The scripts are the intellectual property of Parag and EPFL. Their redistribution under the liberal __GPLv3__ license is with the approval of EPFL, acting through Prof. Marilyne Andersen.
