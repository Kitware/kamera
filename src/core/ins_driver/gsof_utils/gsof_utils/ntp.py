import shlex
import re
import subprocess

PAT_STRATUM = re.compile(r"(?<=stratum )(?P<stratum>\d+)")
PAT_OFFSET = re.compile(r"(?<=offset )(?P<offset>\-?[\.\d]+)(?= sec)")
PAT_NOSERVER = re.compile(r"(?P<noserver>no server)")


class NTPError(Exception):
    pass


def ntp_query(addr, count=1, graceful=False, verbose=False):
    cmd = "ntpdate -q -p{count} {addr}".format(count=count, addr=addr)
    if verbose:
        print(cmd)
    proc = subprocess.Popen(shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = proc.communicate()
    out = out.decode()
    err = err.decode().strip()
    if err:
        mat_noserver = re.search(PAT_NOSERVER, err)
        ntp_up = mat_noserver is None
        if not graceful:
            if mat_noserver:
                raise NTPError("Server not found: {}".format(addr))
            raise RuntimeError("Subprocess failed: [{}]\nError: {}".format(cmd, err))
        stratum = 0
        offset = 0
    else:

        mat_stratum = re.search(PAT_STRATUM, out)
        mat_offset = re.search(PAT_OFFSET, out)
        if mat_stratum is None or mat_offset is None:
            raise ValueError("Unable to parse NTP output: {}".format(out))

        stratum = int(mat_stratum.groupdict().get('stratum', 0))
        offset = float(mat_offset.groupdict().get('offset', 0))
        ntp_up = True

    ntp_valid = ntp_up and 0 < stratum < 16
    if verbose:
        print(out + err)
    return {'stratum': stratum, 'offset': offset, 'ntp_up': ntp_up, 'ntp_valid': ntp_valid, 'addr': addr, 'err': err}
