with open("gel_kernels.cu", "r", encoding="latin-1") as f:
    text = f.read()

old_p = '''\tif (isVoidElement(yi, zi, gp)) {
\t\tint gi = get_index(xi, yi, zi, 1, LX, LY);
\t\tpm[gi] = 0.0;
\t\treturn;
\t}
\tint gi = get_index(xi, yi, zi, 1, LX, LY);
\tdouble wmt = wm[gi];'''

new_p = '''\tint gi = get_index(xi, yi, zi, 1, LX, LY);
\tdouble wmt = wm[gi];'''

if old_p in text:
    text = text.replace(old_p, new_p, 1)
    print("calPressureD fixed")
else:
    print("Pattern not found! Check gel_kernels.cu")

with open("gel_kernels.cu", "w", encoding="latin-1") as f:
    f.write(text)
