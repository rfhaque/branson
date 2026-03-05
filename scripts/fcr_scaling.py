import lxml.etree as ET
import numpy as np
import os
import re
import subprocess
import sys

def make_input_file(base_file, transport, storage, algorithm, n_photons):
    r_tree = ET.parse(base_file)
    r_root = r_tree.getroot()

    r_common = r_root.find("common")

    r_common.find("dd_transport_type").text = f"{transport}"
    r_common.find("particle_storage").text = f"{storage}"
    r_common.find("particle_algorithm").text = f"{algorithm}"
    r_common.find("photons").text = f"{n_photons}"


    new_filename = "temp_input.xml"

    r_tree.write(new_filename, pretty_print=True)
    return new_filename

if (len(sys.argv) != 2):
    print("usage: {0} <basic_input_file_name>".format(sys.arv[0]))
    sys.exit()

base_filename = sys.argv[1]

path_to_exe = 
exe_name = "BRANSON"

time_r = re.compile('runtime: (.*?) $')

particle_algorithms = ["EVENT", "HISTORY"]
particle_storage = ["SOA", "AOS"]
use_gpu_transport = ["TRUE", "FALSE"]
dd_transport_type = ["REPLICATED", "PARTICLE_PASS"]

np_list = []
proc_list = []

results_filename = "scaling_results.txt"
f_results = open(results_filename, 'w')

for transport in dd_transport_type:
    for storage in particle_storage:
        for algorithm in particle_algorithms:
            for n_particles in np_list:
                for p in proc_list:
                    times = []

                    temp_input_file = make_input_file(base_filename, transport, storage, algorithm, n_particles)

                    for s in range(samples):
                        time = 0.0
                        output_file = "temp_output.txt"
                        subprocess.call(, shell=True)
                        
                        f_out = open(output_file, 'r')
                        for line in f_out:
                            if (time_r.search(line)):
                                time = float(time_r.findall(line[0])

                        times.append(time)

                    runtime = np.average(times)
                    stdev_runtime = np.std(times)
                    f_results.write()

f_results.close()
