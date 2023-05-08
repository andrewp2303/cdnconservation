# CS 262 Final Project - Adarsh Hiremath, Andrew Palacci, and Artemas Radik

This file contains code that calculates the optimal distribution of surrogate servers in a Content Delivery Network (CDN) to minimize energy consumption. It does so by modeling the energy consumption of the network based on various factors such as the number of surrogate servers, the size of the cache, and the hit rate of the content. The code also generates plots that show the relationship between the number of surrogate servers and the energy consumption of the network.

## Dependencies 
- Python 3 
- numpy 
- matplotlib 

## Usage 
The program can be run using the command `python conservation.py <out.png>` to generate a plot of the energy consumption versus the number of surrogate servers. The output file name must be specified as an argument. Optionally, the command `python conservation.py <out.png> pareto` can be used to generate a plot showing the relationship between the Pareto alpha parameter and the energy consumption of the network.

## Class 
The `Opt` class contains all the variables needed for energy consumption calculation. The class is initialized with a single parameter, `S`, which represents the number of surrogate servers in the network. Other parameters that can be modified include the cache size, the number of modifications to content, the number of requests for content, and the hit rate for content.

## Functions 
The program contains several functions that calculate the energy consumption of the network. These include:
- `E_storage(opt)`: Calculates the CDN storage energy consumption, in Joules.
- `E_server(opt)`: Calculates the CDN server energy consumption, in Joules.
- `E_synch(opt)`: Calculates the CDN synchronization energy consumption, in Joules.
- `hgeom_pmf(k, w, b, n)`: PMF using HGeom(w, b, n) applied to k.
- `E_tran(opt)`: Calculates CDN transmission energy consumption, in Joules.
- `total_energy(opt)`: Calculates the energy consumption of the network, in Joules.
- `pareto_P_hit(alpha, S_c, M)`: Calculates the Pareto hit rate given the values.

## Main function 
The main function generates a plot of the energy consumption versus the number of surrogate servers. It does so by iterating over a range of values for the number of surrogate servers and calculating the energy consumption for each value. The resulting data is then plotted using the matplotlib library.

## Pareto function 
The `pareto_plot` function generates a plot showing the relationship between the Pareto alpha parameter and the energy consumption of the network. It does so by iterating over a range of values for the Pareto alpha parameter and calculating the energy consumption for each value. The resulting data is then plotted using the matplotlib library.

## Plots 
The program generates two plots:
- A plot showing the relationship between the number of surrogate servers and the energy consumption of the network. 
- A plot showing the relationship between the Pareto alpha parameter and the energy consumption of the network. 

## Conclusion 
In summary, the `conservation.py` file contains code that models the energy consumption of a CDN based on various parameters and generates plots that show the relationship between the number of surrogate servers, Pareto alpha parameter and the energy consumption of the network. The program can be modified to include more accurate data if needed.
