import json
import argparse
import math
import logging

NINETY_DEG_X_ROTATION = [math.sqrt(0.5), 0, 0, math.sqrt(0.5)]

def reorient_scene(gltf_data):
    if not gltf_data.get('scenes') or not gltf_data['scenes'][0].get('nodes'):
        logging.warning("No valid scene or nodes found in the glTF file.")
        return gltf_data

    scene = gltf_data['scenes'][0]
    
    if len(scene['nodes']) == 1:
        root_node = gltf_data['nodes'][scene['nodes'][0]]
        if 'rotation' in root_node:
            logging.info("Applying rotation to existing root node.")
            root_node['rotation'] = multiply_quaternions(root_node['rotation'], NINETY_DEG_X_ROTATION)
        else:
            root_node['rotation'] = NINETY_DEG_X_ROTATION
    else:
        logging.info("Creating new root node for reorientation.")
        new_root = {
            "name": "ReorientedRoot",
            "rotation": NINETY_DEG_X_ROTATION,
            "children": scene['nodes']
        }
        scene['nodes'] = [len(gltf_data['nodes'])]
        gltf_data['nodes'].append(new_root)

    return gltf_data

def multiply_quaternions(q1, q2):
    return [
        q1[0]*q2[0] - q1[1]*q2[1] - q1[2]*q2[2] - q1[3]*q2[3],
        q1[0]*q2[1] + q1[1]*q2[0] + q1[2]*q2[3] - q1[3]*q2[2],
        q1[0]*q2[2] - q1[1]*q2[3] + q1[2]*q2[0] + q1[3]*q2[1],
        q1[0]*q2[3] + q1[1]*q2[2] - q1[2]*q2[1] + q1[3]*q2[0]
    ]

def modify_gltf(input_file, output_file, metallic_factor, roughness_factor, override, reorient):
    logging.info(f"Reading input file: {input_file}")
    try:
        with open(input_file, 'r') as f:
            gltf_data = json.load(f)
    except IOError as e:
        logging.error(f"Error reading input file: {e}")
        return

    if reorient:
        logging.info("Reorienting scene...")
        gltf_data = reorient_scene(gltf_data)

    if 'materials' in gltf_data:
        logging.info("Modifying materials...")
        for material in gltf_data['materials']:
            pbr = material.setdefault('pbrMetallicRoughness', {})
            if 'metallicFactor' not in pbr or override:
                pbr['metallicFactor'] = max(0, min(1, metallic_factor))
            if 'roughnessFactor' not in pbr or override:
                pbr['roughnessFactor'] = max(0, min(1, roughness_factor))
    else:
        logging.warning("No materials found in the glTF file.")

    logging.info(f"Writing output file: {output_file}")
    try:
        with open(output_file, 'w') as f:
            json.dump(gltf_data, f, indent=2)
        logging.info("Modification complete.")
    except IOError as e:
        logging.error(f"Error writing output file: {e}")

def main():
    parser = argparse.ArgumentParser(description="Modify materials in a GLTF file and optionally reorient the scene.")
    parser.add_argument("input_file", help="Path to the input GLTF file")
    parser.add_argument("output_file", help="Path to save the modified GLTF file")
    parser.add_argument("--metallic", type=float, default=0.1, help="Set the metallic factor (default: 0.1)")
    parser.add_argument("--roughness", type=float, default=0.1, help="Set the roughness factor (default: 0.1)")
    parser.add_argument("--override", action="store_true", help="Override existing material values if set")
    parser.add_argument("--reorient", action="store_true", help="Reorient the scene from Z-up to Y-up")
    parser.add_argument("--log", choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], default='INFO', help="Set the logging level")

    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log), format='%(levelname)s: %(message)s')

    modify_gltf(args.input_file, args.output_file, args.metallic, args.roughness, args.override, args.reorient)

if __name__ == "__main__":
    main()