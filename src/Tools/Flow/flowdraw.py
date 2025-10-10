#!/usr/bin/env python3
import sys, tempfile, os
from xml.dom.minidom import parse


def clean(s):
    s = s.replace("-", "_")
    s = s.replace("~", "_")
    return s


def main(argv):
    if len(argv) < 2:
        usage(argv[0])

    dom = parse(argv[1])
    nodes = dom.getElementsByTagName("node")
    links = dom.getElementsByTagName("link")
    inputs = dom.getElementsByTagName("in")
    outputs = dom.getElementsByTagName("out")
    params = dom.getElementsByTagName("param")
    network = dom.getElementsByTagName("network")[0]

    out, outname = tempfile.mkstemp()
    out = open(outname, "w")
    print("digraph flow {", file=out)

    att = lambda n, a: n.getAttribute(a)

    txt = ""
    netname = att(network, "name")
    if not netname:
        netname = "network"

    for i in inputs + outputs:
        i = att(i, "name")
        print(
            """%s [shape=plaintext
			label = "%s:%s"];"""
            % (i, netname, i),
            file=out,
        )

    for n in nodes:
        name = att(n, "name")
        atts = n.attributes
        inp = ""

        label = (
            "<FONT FACE='svg'><TABLE CELLSPACING='0'>%s<TR><TD COLSPAN='2'>%s</TD></TR>"
            % (inp, name)
        )
        for i in range(len(atts)):
            key = atts.item(i).name
            value = att(n, key)
            # 			value = clean(value)
            if key == "name":
                continue
            if key == "filter":
                if value[-5:] == ".flow":
                    value = "<FONT COLOR='RED'>%s</FONT>" % value
                elif value[0] != "$":
                    value = "<FONT COLOR='DARKGREEN'>%s</FONT>" % value
            if value[0] == "$":
                value = "<FONT COLOR='BLUE'>%s</FONT>" % value

            label += "<TR><TD ALIGN='LEFT'>%s</TD><TD ALIGN='LEFT'>%s</TD></TR>" % (
                key,
                value,
            )

        label += "</TABLE></FONT>"
        name = clean(name)

        print(
            """%s [shape=plaintext
		label = <%s>
		];"""
            % (name, label),
            file=out,
        )

    for l in links:

        fr = att(l, "from")
        to = att(l, "to")

        fr = clean(fr)
        to = clean(to)

        if fr.startswith(netname + ":"):
            fr = fr.split(":")[1]
        if to.startswith(netname + ":"):
            to = to.split(":")[1]

        label = ""
        if ":" in fr or ":" in to:
            port1 = ""
            port2 = ""
            if ":" in fr:
                port1 = fr.split(":")[1]
            if ":" in to:
                port2 = to.split(":")[1]
            label = ' [label="%s->%s", fontname="svg"]' % (port1, port2)

        print("%s -> %s %s;" % (fr, to, label), file=out)

    print("}", file=out)
    # print "dot -Tpng < %s > %s.png; eog %s.png " % (outname, outname, outname)
    out.close()

    if len(argv) == 3:
        pic = argv[2]
    else:
        pic = "%s.png" % outname

    os.system("dot -Tpng < %s > %s" % (outname, pic))
    print("Plot written to %s" % pic)


def usage(prog):
    print("USAGE: %s file.flow [plot.png]" % prog)
    sys.exit(-1)


if __name__ == "__main__":
    main(sys.argv)
