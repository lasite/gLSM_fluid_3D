with open("sim_gelonly_tube.cpp", "r") as f:
    code = f.read()
import re
new_code = code.replace('            int test_gi = tube->get_index(1, tube->m_gelNodeGrid.y/2, tube->m_gelNodeGrid.z/2, 1);',
'''            int test_gi = tube->get_index(1, 1, tube->m_gelNodeGrid.z/2, 1);
            int void_gi = tube->get_index(1, tube->m_gelNodeGrid.y/2, tube->m_gelNodeGrid.z/2, 1);
''').replace('printf("iter=%6d  Gel V_max = %.5f  um[pulse]=%.5f vm[pulse]=%.5f\\n", iter, max_v, tube->m_hum[test_gi], tube->m_hvm[test_gi]);',
'''printf("iter=%6d  Gel V_max = %.5f  um[solid]=%.5f vm[solid]=%.5f | um[void]=%.5f vm[void]=%.5f\\n", iter, max_v, tube->m_hum[test_gi], tube->m_hvm[test_gi], tube->m_hum[void_gi], tube->m_hvm[void_gi]);''')
with open("sim_gelonly_tube.cpp", "w") as f:
    f.write(new_code)
