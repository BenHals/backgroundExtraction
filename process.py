import seg as seg
import createbg as cb
import gen3d
import genunity
import argparse


if __name__ == "__main__":
    # Extract and set command line arguments for parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--blocksize", type=float, help="enter some quality limit",
                        nargs='?', default=0.035)
    parser.add_argument("--multi", type=float, help="enter some quality limit",
                        nargs='?', default=1.95)
    parser.add_argument("--texthresh", type=float, help="enter some quality limit",
                        nargs='?', default=0.1)
    parser.add_argument("--colthresh", type=float, help="enter some quality limit",
                        nargs='?', default=0.4)
    parser.add_argument("--file", help="enter some quality limit",
                        nargs='?', default='miro')
    parser.add_argument("--unity", help="enter some quality limit",
                        nargs='?', default='D:/PROGFILES/unity/Editor/Unity.exe')
                        # nargs='?', default='D:/programs/unity/Editor/Unity.exe')
    parser.add_argument("--detail", help="enter some quality limit", action='store_true', default=False)
    parser.add_argument("--sbsize", type=float, default=512)
    args = parser.parse_args()
    BLOCK_SIZE = args.blocksize 
    COLTHRESH = args.colthresh
    TEXTHRESH = args.texthresh
    FILE = args.file
    DETAIL = args.detail
    MULTI = args.multi
    SBSIZE = args.sbsize
    UNITY = args.unity

    seg.seg(BLOCK_SIZE, COLTHRESH, TEXTHRESH, FILE, DETAIL, MULTI)
    cb.create_bg(BLOCK_SIZE, COLTHRESH, TEXTHRESH, FILE, DETAIL, MULTI, SBSIZE)
    #gen3d.gen_3d(BLOCK_SIZE, COLTHRESH, TEXTHRESH, FILE, DETAIL, MULTI, SBSIZE)
    genunity.gen_unity(BLOCK_SIZE, COLTHRESH, TEXTHRESH, FILE, DETAIL, MULTI, SBSIZE, UNITY)


