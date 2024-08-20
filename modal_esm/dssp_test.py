from modal import Image, App, method

def build_image():
    #https://github.com/biotite-dev/biotite/issues/622
    return (
        Image.debian_slim()
        .run_commands(
            "apt-get update && apt-get install -y git build-essential wget zlib1g-dev libboost-all-dev",
            "wget https://github.com/Kitware/CMake/releases/download/v3.26.4/cmake-3.26.4-linux-x86_64.sh",
            "chmod +x cmake-3.26.4-linux-x86_64.sh",
            "./cmake-3.26.4-linux-x86_64.sh --skip-license --prefix=/usr/local",
            "rm cmake-3.26.4-linux-x86_64.sh",
            "git clone https://github.com/cmbi/dssp.git",
        )
        .run_commands(
            "cd dssp && ./autogen.sh && ./configure && make && make install",
            "ldconfig",  # Update shared library cache
            "ln -s /usr/local/bin/mkdssp /usr/local/bin/dssp",  # Create symlink for compatibility
            "cd .. && rm -rf dssp"
        )
        .pip_install("biotite")
    )

image = build_image()
app = App("dssp-test", image=image)

@app.cls()
class DSSPTest:
    @method()
    def test_dssp(self):
        import subprocess
        import biotite.application.dssp as dssp
        import biotite.structure as struc
        import biotite.structure.io.pdbx as pdbx
        import biotite.database.rcsb as rcsb
        import numpy as np
        from os.path import join

        try:
            # Test DSSP installation
            result = subprocess.run(["mkdssp", "--version"], capture_output=True, text=True, check=True)
            print(f"DSSP version: {result.stdout.strip()}")

            # Define the PDB ID and chain ID
            pdb_id = "1YFP"
            chain_id = "A"

            # Fetch the structure from RCSB
            file_path = rcsb.fetch(pdb_id, "pdbx", "./")
            structure = pdbx.get_structure(pdbx.PDBxFile.read(file_path), model=1)
            
            # Filter for the specified chain
            atoms = structure[struc.filter_amino_acids(structure) & (structure.chain_id == chain_id)]
            sse = dssp.DsspApp.annotate_sse(atoms)
            
            assert np.all(np.isin(sse, ["C", "H", "B", "E", "G", "I", "T", "S"]))
            
            
            print("Biotite DSSP test passed successfully!")
            return True
        except Exception as e:
            print(f"Error running DSSP test: {e}")
            return False

@app.local_entrypoint()
def main():
    dssp_test = DSSPTest()
    success = dssp_test.test_dssp.remote()
    if success:
        print("DSSP installed and running successfully!")
    else:
        print("Failed to run DSSP.")

if __name__ == "__main__":
    main()