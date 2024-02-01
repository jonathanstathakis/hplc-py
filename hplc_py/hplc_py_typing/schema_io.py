from hplc_py import ROOT
import pandas as pd
import os
import warnings
import pandera as pa

class CacheSchemaValidate:
    """
    Compare an input frame against a selected cached schema.
    This both handles the cache directory path and loads the cache file object as
    `SchemaCacheFile` for io behavior.
    
    TODO:
        - [ ] add error handling
        - [ ] flesh out cache dir handling class - provide information about the cache.
        - [ ] flesh out cache file class - provide information about the cache file
        - [ ] add logic regarding 'staleness' of schema.
    """
    
    def __init__(
        self,
        
    ):
        self._rootpath = ROOT
        self._cache_dir_path = os.path.join(ROOT, "schema_cache")
        self._check_cache_dirpath_exists()
      
    def get_schema(
        self,
        df,
        dset_name: str,
        schema_name: str,
    ):
        self.dset_name = dset_name
        self.schema_name = schema_name
        
        self.sch = self.get_cached_schema(df, dset_name, schema_name)
        
        return self.sch
    
    def get_cached_schema(self, df, dset_name: str, schema_name: str):
        
        sc = SchemaCacheFile(self._cache_dir_path, dset_name, schema_name)
        
        sc.io_cache(df)
        
        return sc.sch
        
        
    def _check_cache_dirpath_exists(
        self,
    )->None:
        """
        Create the filetree to the cache from the project root if none exists
        """
        if not os.path.isdir(self._cache_dir_path):
            ip = input(f"no cache dir found at {self._cache_dir_path}\n\nCreate now? (y/n):")
            
            if ip=='y':
                os.makedirs(self._cache_dir_path)
            else:
                raise RuntimeError("No cache dir detected, cannot read cache")
        return None
        

class SchemaCacheFile:
    """
    Handle IO of inferred schema to and from YAML formats. To be used within testing
    environments to cache schema.
    
    For dataset specific checking. Will accept the frame and a dataset name.
    
    1. accept a frame
    2. create the schema.
    3. check the cache for files matching the intended destination
    4. if present, use
    5. if not present, create and warn user.
    """
    def __init__(self, path: str, dset_name: str, schema_name: str):
        self._cache_dir_path = path
        self._dset_name = dset_name
        self._schema_name = schema_name
        
        self._set_cache_file_path()
        
    def _set_cache_file_path(
        self
    ):
        self._cache_file_path = os.path.join(self._cache_dir_path, self._dset_name+"_"+self._schema_name+".yml")

    def io_cache(self, df: pd.DataFrame):
        """
        1. set the io path
        2. check if dir exists
            1. if not, warn user and create path
        2. check if anything present at path
            1. if not, warn user, create cache at path, return nothing.
            2. if yes, load schema and return
        """
        self._df = df
    
        self._io_schema_cache_file(self._cache_file_path)

    def _io_schema_cache_file(self, path: str):
        """
        1. check `self.path` to see if object present.
            1. If present, attempt to read as schema.
                1. if fails to parse, let user know and end session.
                2. if successful, return schema obj.
            2. if not present, ask user if to create
                1. create yaml object
                2. write yaml to path.
                3. warn user about write location.
                3. return nothing.
        2. return
        """
        if not os.path.isfile(path):
            ip = input(f"No schema cache file found at {path}\n Create? (y/n):")
            if ip == 'y':
                self._write_cache(path)
            else:
                raise RuntimeError("no cache file detected at path, cannot read cache")
        else:
            self._read_cache(path)
            
        return None

    def _df_to_yaml(self):
        self.sch = pa.infer_schema(self._df)
        self._yaml = self.sch.to_yaml()
        
    def _write_cache(self, path: str):
        """
        Write the df schema yaml to file
        """
        self._df_to_yaml()
        with open(path, 'w') as f:
            f.write(self._yaml)
            wrn = f"Written schema as yaml to {path}"
        warnings.warn(wrn)
        return None
    
    def _read_cache(
        self,
        path: str,
    ):
        """
        Convert the cache to a schema object
        """
        from pandera import io
        try:
            self.sch = io.from_yaml(path)
        except pa.errors.SchemaDefinitionError as e:
            e.add_note(f"tried to read schema at {path}")
            raise e

        return None

def main():
    
    df = pd.read_parquet("hplc_py/schema_cache/bounds.parquet")
    cacheval = CacheSchemaValidate()
    sch = cacheval.get_schema(df, dset_name='asschrom',schema_name='bounds')
    
    try:
        sch.validate(df.drop('ub',axis=1), lazy=True)
    except Exception as e:
        raise e
    else:
        print("passed")


if __name__ == "__main__":
    main()