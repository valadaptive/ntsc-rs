import sys

version = sys.argv[1]
version = [int(v) for v in version.split('.')]
if len(version) != 3:
    print('Invalid version format')
    sys.exit(1)
pipl_version = (version[0] << 19) | (version[1] << 15) | (version[2] << 11) | (3 << 9) | 1
print(pipl_version)