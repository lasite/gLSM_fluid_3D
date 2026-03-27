with open("sim_gelonly_tube.cpp", "r") as f:
    code = f.read()
import re
new_code = code.replace('printf("iter=%6d  Gel V_max = %.5f\\n", iter, max_v);',
'''
            cudaMemcpy(tube->m_hum, tube->m_dum, tube->m_numGelElements * sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(tube->m_hvm, tube->m_dvm, tube->m_numGelElements * sizeof(double), cudaMemcpyDeviceToHost);
            int test_gi = tube->get_index(1, tube->m_gelNodeGrid.y/2, tube->m_gelNodeGrid.z/2, 1);
            printf("iter=%6d  Gel V_max = %.5f  um[pulse]=%.5f vm[pulse]=%.5f\\n", iter, max_v, tube->m_hum[test_gi], tube->m_hvm[test_gi]);
            
''')
with open("sim_gelonly_tube.cpp", "w") as f:
    f.write(new_code)
