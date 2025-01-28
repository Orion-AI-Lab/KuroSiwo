import os, sys
import enum
import json
import ciso8601
import yaml
import argparse
import uuid as uuidmod
from pathlib import Path
from collections import OrderedDict
import numpy as np
import pandas as pd
import rioxarray as rio
import geopandas as gp
from compress_pickle import load, dump


def join(loader, node):
    seq = loader.construct_sequence(node)
    return "".join([str(i) for i in seq])


yaml.add_constructor("!join", join)
_F = os.path.abspath(__file__)


def normpath(extpath, basepath=_F):
    if basepath is None:
        return None
    elif os.path.isabs(extpath):
        return extpath
    elif os.path.isfile(basepath):
        return str(os.path.normpath(os.path.join(os.path.dirname(basepath), extpath)))
    else:
        return str(os.path.normpath(os.path.join(basepath, extpath)))


def _parseCFG():
    CFG["DATA_PATH"] = normpath(CFG["DATA_PATH"], _F)
    CFG["CAT_PATH"] = normpath(CFG["CAT_PATH"], _F)
    CFG["REPO_PATH"] = normpath(CFG["REPO_PATH"], _F)
    CFG["CL_ZONES"] = {_["cl_zone"]: _ for _ in CFG["CL_ZONES"]}

    _pa = lambda _: _ | {"cl_name": CFG["CL_ZONES"][_["cl_zone"]]["cl_name"]}
    _pp = dict(
        act_id=str,
        act_region=str,
        ref_date=lambda v: ciso8601.parse_datetime(v),
        aois=lambda l_: {a_["aoi_id"]: _pa(a_) for a_ in l_},
    )
    _pinfo = lambda x: {k: _pp[k](v) for k, v in x.items()}
    CFG["Floods"] = {_pp["act_id"](v["act_id"]): _pinfo(v) for v in CFG["Floods"]}


_CFG = normpath("./catalogue.yaml")
with open(_CFG) as f:
    CFG = yaml.load(f, Loader=yaml.Loader)

_parseCFG()


class GridProduct(object):
    _NODATA = dict(IVH=0.0, IVV=0.0, MNA=0, MLU=3)
    _DTYPE = dict(IVH="float32", IVV="float32", MNA="uint8", MLU="uint8")

    @property
    def nodata(self):
        return self._NODATA[self.pname] if self.pname else None

    @property
    def dtype(self):
        return np.dtype(self._DTYPE[self.pname]) if self.pname else None

    @property
    def ptype(self):
        return ("MS" if self.master else "SL") if self.rank else "MK"

    @property
    def name(self):
        if not self._name:
            aoiid = "{:02d}".format(self.gd.aoiid) if self.gd.aoiid else "NA"
            self._name = f'{self.ptype}{self.rank}_{self.pname}_{self.gd.actid}_{aoiid}_{self.source_date.strftime("%Y%m%d")}'
        return self._name

    @property
    def clz_id(self):
        return self._aoi_meta["cl_zone"]

    @property
    def clz_name(self):
        return self._aoi_meta["cl_name"]

    @property
    def path(self):
        if (not self._path) and self.name:
            self._path = self.base.joinpath(self.name + ".tif")
        return self._path

    def todict(self):
        info = dict(
            name=self.name,
            master=self.master,
            nodata=self.nodata,
            dtype=self.dtype,
            ptype=self.ptype,
            pname=self.pname,
            rank=self.rank,
            source_date=self.source_date,
            clz_id=self.clz_id,
            clz_name=self.clz_name,
            sources=self.sources,
        )
        return info

    def __init__(self, gd=None, base=None, pname=None):
        self._name = None
        self._path = None
        self.gd = gd
        self.base = base
        self.master = gd.master and pname not in ("MLU", "MNA")
        self.pname = pname
        self.rank = 0 if pname in ("MLU", "MNA") else self.gd.crank
        self.source_date = self.gd.source_date
        self.sources = self.gd.s1_ids
        try:
            self._aoi_meta = CFG["Floods"][f"{self.gd.actid}"]["aois"][
                f"{self.gd.aoiid:02d}"
            ]
        except:
            if self.gd.aoiid is None:
                self.gd.aoiid = 999999
            self._aoi_meta = dict(
                aoi_id=f"{self.gd.aoiid:02d}", aoi_name=None, cl_zone=None, cl_name=None
            )

    def __repr__(self):
        return self.name


class CatalogueGrid(object):
    @property
    def valid(self):
        if self._valid is None:
            self._valid = bool(self.gdmaster.gvalid)
        return self._valid

    @property
    def id(self):
        return self.gdmaster.grid_id

    @property
    def inaoi(self):
        return bool(self.gdmaster.aoiid is not None)

    @property
    def aoi(self):
        return self.gdmaster.aoiid

    @property
    def geom(self):
        return self.gdmaster["geom"]

    @property
    def path(self):
        if self.inaoi:
            pargs = [
                str(self.gdmaster["actid"]),
                "{:02d}".format(self.gdmaster["aoiid"]),
                self.gdmaster["grid_id"].hex,
            ]
        else:
            pargs = [
                str(self.gdmaster["actid"]),
                "00",
                self.gdmaster["grid_id"].hex[0:2],
                self.gdmaster["grid_id"].hex,
            ]

        return Path(CFG["REPO_PATH"]).joinpath(*pargs)

    @property
    def sources(self):
        return [i for sl in self.gdf["s1_ids"].tolist() for i in sl]

    def __init__(self, ggdf):
        self.gdf = ggdf
        self.datasets = list()
        self.gdmaster = None
        self.gdslave = list()
        self.info = None
        self._valid = None
        if self.gdf is None or not self.gdf.iloc[0].gvalid:
            self._valid = False
            return

        self.gdmaster = self.gdf.loc[self.gdf["master"] == True].iloc[0]
        self.gdslave = [
            row for _, row in self.gdf.loc[self.gdf["master"] == False].iterrows()
        ]

        self.infopath = self.path.joinpath("info.json")
        for ptype in ("IVH", "IVV"):
            for gd in [
                self.gdmaster,
            ] + self.gdslave:
                self.datasets.append(GridProduct(gd=gd, base=self.path, pname=ptype))

        self.datasets.append(GridProduct(gd=self.gdmaster, base=self.path, pname="MNA"))
        self.clz_id = self.datasets[-1].clz_id
        self.clz_name = self.datasets[-1].clz_name
        if self.inaoi:
            self.datasets.append(
                GridProduct(gd=self.gdmaster, base=self.path, pname="MLU")
            )

        if self.infopath.exists():
            self.info = json.loads(self.infopath.read_text())


class Catalogue(object):
    _schema = {
        "geometry": "Polygon",
        "properties": OrderedDict(
            [
                ("grid_id", "str"),
                ("slavecov", "int"),
                ("mastercov", "int"),
                ("gvalid", "bool"),
                ("pcovered", "float"),
                ("pwater", "float"),
                ("pflood", "float"),
                ("actid", "int"),
                ("flood_date", "datetime"),
                ("aoiid", "int"),
                ("revision", "int"),
                ("version", "int"),
                ("source_date", "date"),
                ("s1_ids", "str"),
                ("master", "bool"),
                ("coverage", "float"),
                ("crank", "int"),
            ]
        ),
    }

    _schema_ext = {
        "geometry": "Polygon",
        "properties": OrderedDict(
            [
                ("grid_id", "str"),
                ("slavecov", "int"),
                ("mastercov", "int"),
                ("gvalid", "bool"),
                ("pcovered", "float"),
                ("pwater", "float"),
                ("pflood", "float"),
                ("actid", "int"),
                ("flood_date", "datetime"),
                ("aoiid", "int"),
                ("revision", "int"),
                ("version", "int"),
                ("source_date", "date"),
                ("s1_ids", "str"),
                ("master", "bool"),
                ("coverage", "float"),
                ("crank", "int"),
                ("exported", "bool"),
                ("shuffle", "int"),
            ]
        ),
    }

    class Schema(str, enum.Enum):
        simple = "simple"
        extended = "extended"

    @property
    def ids(self):
        for id in list(self.gdf.grid_id.unique()):
            yield id

    @property
    def actids(self):
        for id in list(self.gdf.actid.unique()):
            yield id

    @property
    def aoiids(self):
        uv = [
            _
            for _ in list(
                self.gdf.drop_duplicates(["actid", "aoiid"])[
                    ["actid", "aoiid"]
                ].to_records(index=False)
            )
            if _[1] is not None
        ]
        for uv_ in uv:
            yield uv_

    def filter(
        self, pcovered=None, ppermwater=None, pflooded=None, pwater=None, coverage="ALL"
    ):
        gdf_ = self.gdf
        gdf_ = gdf_[gdf_.gvalid == True]
        if pcovered:
            gdf_ = gdf_[gdf_.pcovered.between(*pcovered)]
        if ppermwater:
            gdf_ = gdf_[gdf_.pwater.between(*ppermwater)]
        if pflooded:
            gdf_ = gdf_[gdf_.pflood.between(*pflooded)]
        if pwater:
            gdf_ = gdf_[(gdf_.pwater + gdf_.pflood).between(*pwater)]

        gdf_.sort_values(
            by=["actid", "aoiid", "grid_id", "master", "crank"],
            ascending=True,
            inplace=True,
        )
        fcat = Catalogue(gdf_, schema=self.Schema.extended)
        stats_ = dict(
            records=len(fcat),
            activations=len(list(fcat.actids)),
            actids=list(fcat.actids),
            aois=len(list(fcat.aoiids)),
            grids=len(list(fcat.ids)),
            coverage=coverage,
        )

        if coverage == "AOI":
            statsext_ = dict(
                mean_pcovered=fcat.gdf[fcat.gdf["pcovered"] != None]["pcovered"].mean(),
                mean_ppermwater=fcat.gdf[fcat.gdf["pwater"] != None]["pwater"].mean(),
                mean_pflooded=fcat.gdf[fcat.gdf["pflood"] != None]["pflood"].mean(),
            )
            stats_ = stats_ | statsext_
        fcat.stats = stats_
        return fcat

    def grid(
        self,
        id,
    ):
        gridds = self.gdf.loc[self.gdf["grid_id"] == id]
        return CatalogueGrid(gridds)

    def write(self, path):
        gdf = self.gdf.copy()
        gdf["grid_id"] = self.gdf["grid_id"].apply(
            lambda c: str(c) if c is not None else c
        )
        gdf["s1_ids"] = self.gdf["s1_ids"].apply(
            lambda c: json.dumps([str(_) for _ in c]) if c is not None else c
        )
        self.schema_ = self._schema_ext if "exported" in gdf.columns else self._schema
        gdf.sort_values(
            by=["actid", "aoiid", "grid_id", "master", "crank"],
            ascending=True,
            inplace=True,
        )
        gdf.to_file(path, schema=self.schema_, driver="GPKG")

    def read(self, path, schema=None):
        self.schema_ = self._schemamap[schema.name] if schema else self.schema_

        gdf = gp.read_file(path, schemas=self.schema_)
        gdf["grid_id"] = gdf["grid_id"].apply(
            lambda c: uuidmod.UUID(c) if c is not None else c
        )
        gdf["s1_ids"] = gdf["s1_ids"].apply(
            lambda c: [uuidmod.UUID(_) for _ in json.loads(c)] if c is not None else c
        )
        gdf["flood_date"] = gdf["flood_date"].apply(
            lambda c: ciso8601.parse_datetime(c) if c is not None else c
        )
        gdf["source_date"] = gdf["source_date"].apply(
            lambda c: ciso8601.parse_datetime(c).date() if c is not None else c
        )
        gdf["aoiid"] = gdf["aoiid"].astype("Int64")
        gdf["revision"] = gdf["revision"].astype("Int64")
        gdf["version"] = gdf["version"].astype("Int64")
        gdf = gdf.replace({np.nan: None})
        gdf = gdf.rename(columns={"geometry": "geom"}).set_geometry("geom")

        if "exported" not in gdf.columns:
            gdf["exported"] = False
        gdf["exported"] = gdf["exported"].astype("bool")
        if not "shuffle" in gdf.columns:
            gdf["shuffle"] = 0
        gdf["shuffle"] = gdf["shuffle"].astype("Int64")

        gdf.sort_values(
            by=["actid", "aoiid", "grid_id", "master", "crank"],
            ascending=True,
            inplace=True,
        )
        self.gdf = gdf

    def __init__(self, gdf=None, schema=None):
        self.gdf = gdf
        self._schemamap = {
            k.name: v
            for k, v in zip(
                (self.Schema.simple, self.Schema.extended),
                (self._schema, self._schema_ext),
            )
        }
        self.schema_ = self._schemamap[schema.name] if schema else self._schema
        self.stats = None

    def __len__(self):
        if self.gdf is not None:
            return len(self.gdf)
        return 0


_parser = None


def create_parser():
    def oper(args):
        cpath = (
            Path(CFG["CAT_PATH"])
            if not args.catalogue_path
            else Path(args.catalogue_path)
        )
        dpath = Path(CFG["REPO_PATH"]) if not args.data_path else Path(args.data_path)

        if not cpath.exists() and cpath.is_file():
            raise FileNotFoundError(
                f"Provided catalogue dataset {cpath.as_posix()} does not exist"
            )
        if not dpath.exists() and dpath.is_file():
            raise FileNotFoundError(
                f"Provided datasets repository {dpath.as_posix()} does not exist"
            )

        onlyinfo = args.info
        for f in ("pcovered", "ppermwater", "pflooded", "pwater"):
            v = getattr(args, f)
            if v:
                v_ = json.loads(v)
                assert (
                    isinstance(v_, list)
                    and len(v_) == 2
                    and 0 <= v_[0] <= 100
                    and 0 <= v_[1] <= 100
                ), "Range value error, use template e.g: [0,100]"
                setattr(args, f, v_)

        cat = Catalogue(schema=Catalogue.Schema.extended)
        cat.read(cpath)
        if args.coverage != "AOI":
            args.pflooded = None
            args.ppermwater = None
            args.pwater = None
        fcat = cat.filter(
            pcovered=args.pcovered,
            ppermwater=args.ppermwater,
            pflooded=args.pflooded,
            pwater=args.pwater,
            coverage=args.coverage,
        )
        stats = json.dumps(fcat.stats, default=str)
        print("Stats for query:")
        print(stats)

        if onlyinfo:
            return

        grid_dict = {}
        invalid_number = 0
        for i, id in enumerate(fcat.ids):
            print(f"Grid {i+1}/{len(cat)}: {id}")
            gridd = fcat.grid(id)
            if not gridd.valid:
                print(f"Invalid Grid: {id}")
                invalid_number += 1
                continue
            if "999999" in str(gridd.path):
                continue
                """aois_path = str(gridd.path).split("999999")[0]
                aois_list = os.listdir(aois_path)
                for aoi_ in aois_list:
                    if not os.path.isdir(os.path.join(aois_path, aoi_)):
                        continue
                    ids = os.listdir(os.path.join(aois_path, aoi_))
                    if gridd.id in ids:
                        gridd.path = str(gridd.path).replace("999999", str(aoi_))
                        break"""
            relative_path = str(gridd.path).split("/data/")[1]
            
            if gridd.info is None:
                print("None info for grid: ", gridd)
                continue
                exit(2)
            grid_dict[gridd.id.hex] = {
                "path": relative_path,
                "info": gridd.info,
                "clz": gridd.clz_id,
                "clz_name": gridd.clz_name,
            }
        print('Num of invalid grids: ', invalid_number)
        print('Num of valid grids: ', len(grid_dict))
        pickle_path = CFG["PICKLE_PATH"]

        Path(os.path.dirname(pickle_path)).mkdir(parents=True, exist_ok=True)
        with open(pickle_path, "wb") as file:
            dump(grid_dict, file)
            print("Saved pickle")

    _parser = argparse.ArgumentParser(
        prog="NNFloods_oper", description="NNFloods Catalogue Operations"
    )
    _parser.set_defaults(func=oper)

    _parser.add_argument("--version", action="version", version="1.0.0")
    _parser.add_argument(
        "--catalogue-path", type=str, help="Merged catalogues path or <CAT_PATH>@CONF"
    )
    _parser.add_argument(
        "--data-path", type=str, help="Root datasets path or <REPO_PATH>@CONF"
    )

    fg = _parser.add_argument_group(
        title="Filters Group", description="Define Filters with <AND> logic"
    )
    fg.add_argument(
        "-a",
        "--activations",
        nargs="+",
        required=False,
        default="ALL",
        type=str,
        action="store",
        choices=list(CFG["Floods"].keys())
        + [
            "ALL",
        ],
        help="Filter activations, DEFAULT: %(default)s",
    )
    fg.add_argument(
        "-c",
        "--coverage",
        nargs="?",
        required=False,
        default="ALL",
        type=str,
        action="store",
        choices=["ACV", "AOI", "ALL"],
        help="Filter by coverage, DEFAULT: %(default)s, ACV-> Only out of AOI Areas, AOI-> only on AOI areas, ALL-> Both areas",
    )
    fg.add_argument(
        "-pv",
        "--pcovered",
        type=str,
        required=False,
        default="[40,100]",
        help="Valid percentage of grid, DEFAULT: %(default)s",
    )
    fg.add_argument(
        "-pw",
        "--ppermwater",
        type=str,
        required=False,
        default="[0,100]",
        help="Percentage of grid with permanent water, DEFAULT: %(default)s",
    )
    fg.add_argument(
        "-pf",
        "--pflooded",
        type=str,
        required=False,
        default="[0,100]",
        help="Percentage of grid with flooded water, DEFAULT: %(default)s",
    )
    fg.add_argument(
        "-pW",
        "--pwater",
        type=str,
        required=False,
        default="[0,100]",
        help="Total percentage of grid water (Permanent+Flooded), DEFAULT: %(default)s",
    )

    _parser.add_argument(
        "-i", "--info", action="store_true", help="Display info on filtered data"
    )

    return _parser


parser = create_parser()


def main(argv):
    print("NNFloods Catalogue OPERATIONS")

    args = parser.parse_args()
    if not bool(args.__dict__):
        raise KeyError("No arguments supplied")
    try:
        args.func(args)
    except Exception as e:
        raise RuntimeError(e)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
