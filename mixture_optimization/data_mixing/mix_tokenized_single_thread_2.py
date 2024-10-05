import argparse
from collections import defaultdict
import json
import logging
import os
import random
import time
import webdataset as wds
from webdataset import ShardWriter
from typing import List
import sys
import numpy as np
import math


def write_to_shard(chunks, shard_writer):
    for idx, chunk in enumerate(chunks):
        shard_writer.write({"__key__": f"{idx:012d}", "txt": str(chunk)})


def linear_access_shard_reader(shard_path, logger):
    assert shard_path.endswith(".tar"), "Shard should be a tar file according to the webdataset"
    idx = -1
    dataset = wds.WebDataset(shard_path)
    dataset_iter = iter(dataset)

    def get_next_element(i):
        nonlocal idx
        assert i >= 0, "Index should be greater than or equal to 0"
        assert i > idx, "Index should be greater than the current index"
        try:
            while idx < i:
                idx += 1
                el = next(dataset_iter)
            element = el["txt"]
            return element.decode("utf-8")
        except Exception as e:
            logger.error(f"Some issue appeared while reading shard {shard_path} at index {i}. Error: {str(e)}")
            idx = i
            return None

    return get_next_element




def mix_tokenized_data(domains: List[str], manifests: List[str], mixing_weights: List[float], output_dir: str, output_token_count: int, chunk_size:int = 2048, shard_size=2048, oversample_factor=1.0, log_path: str = None, log_level: str = "INFO", shard_selection_multiplier=None, seed=None):
    
    
    
    logger = logging.getLogger("data_mixing")
    logger.setLevel(log_level)
    # add stream handler to console
    handler = logging.StreamHandler(stream=sys.stdout)
    logger.addHandler(handler)
    if log_path:
        handler = logging.FileHandler(log_path)
    logger.addHandler(handler)

    # Sanity checks
    assert len(manifests) == len(mixing_weights) == len(domains), "Number of manifests and mixing weights should be the same"
    start_time = time.time()

    # Setup random number generator
    if seed:
        logger.info(f"Overriding seed to {seed}")
        original_random_state = random.getstate()
        random.seed(seed)

    # Normalize mixing weights
    mixing_weights = [weight / sum(mixing_weights) for weight in mixing_weights]

    logger.info("Input domains and desired mixing factors")
    for domain, weight in zip(domains, mixing_weights):
        logger.info(f"{domain}: {weight}")
   
    # output token count must be a multipe of SHARD_SIZE * chunk_size
    output_token_count_extended = oversample_factor * output_token_count +  shard_size * chunk_size # adding one extra shard to make sure we can ommit last one, second one because we somehow always end up with too few tokens
    logger.info(f"Output token count extended to {output_token_count_extended}. This represents {output_token_count_extended / chunk_size / shard_size} shards")

    # calculate no of sequences in each domain
    desired_domain_sequence_counts = {}
    for domain in domains:
        desired_domain_sequence_counts[domain] = math.ceil(output_token_count_extended * mixing_weights[domains.index(domain)] / chunk_size)

    # Load all manifests
    domain_chunk_counts = defaultdict(list)
    for domain, manifest_path in zip(domains, manifests):
        with open(manifest_path, "r") as f:
            for line in f:
                line = json.loads(line)
                shard_name = line["shard"]
                num_sequences = line["num_sequences"]
                manifest_dir_path = os.path.dirname(manifest_path)
                full_shard_path = os.path.join(manifest_dir_path, f"{shard_name}.tar")
                domain_chunk_counts[domain].append((num_sequences, full_shard_path))
    
    # If shard selection multiplier is not none, we will not include all shards in the mixing onlt he required number extended by the multiplier
    if shard_selection_multiplier is not None:
        logger.info(f"Shard selection multiplier is set to {shard_selection_multiplier}. Only a fraction of shards will be included in the mixing")
        for domain, desired_domain_counts in desired_domain_sequence_counts.items():
            no_required_shards = math.ceil(desired_domain_counts / shard_size * shard_selection_multiplier)
            no_domains_original = len(domain_chunk_counts[domain])
            # Edge case that we cannot oversample as planned
            if no_required_shards > no_domains_original:
                logger.info(f"Oversample factor too large. Using all available shards for {domain}. This is {no_domains_original} shards")
                no_required_shards = no_domains_original
                min_shards_required = math.ceil(desired_domain_counts / shard_size)
                if min_shards_required > no_domains_original:
                    logger.error(f"Insufficient data for {domain}. Required {min_shards_required} shards, but only {no_domains_original} are available")
                    sys.exit(1)
            logger.info(f"New shard number for {domain} is {no_required_shards}. This is down from {no_domains_original} shards")
            sample_idx = np.random.choice(no_domains_original, no_required_shards, replace=False)
            domain_chunk_counts[domain] = [domain_chunk_counts[domain][idx] for idx in sample_idx]

    domain_sampling_shard_list = []
    shard_sampling_locations_dict = defaultdict(list)
    data_readers = {}
    for domain, chunk_counts in domain_chunk_counts.items():
        # for each domain extract no of sequences and sample random sequences from them
        total_chunk_counts = sum([count for count, _ in chunk_counts])
        print(f"Domain {domain}, total chunks {total_chunk_counts} desired count { desired_domain_sequence_counts[domain]}")
        domain_sampling_locations = np.random.choice(total_chunk_counts, desired_domain_sequence_counts[domain], replace=False)
        # within each shard sequences must be in order
        domain_sampling_locations.sort()

        shard_idx = 0
        location_counter = 0
        for location in domain_sampling_locations:
            count, shard_path = chunk_counts[shard_idx]
            # current location lies within this shard
            if location < location_counter + count:
                domain_sampling_shard_list.append((domain, shard_path)) # add domain and shard path to list to later random sample one
                shard_sampling_locations_dict[shard_path].append(location - location_counter) # to guarantee sequential reads per shard, keep track of incresing within shard locations
                # for each shard create one reader
                if shard_path not in data_readers:
                    data_readers[shard_path] = linear_access_shard_reader(shard_path, logger=logger)
            else:
                location_counter += count
                shard_idx += 1
    
    # Randomize the order of the domain_sampling_locations_list
    # This randomizes sapling up to a order within a shard, i.e. we randomly sample a shard
    # within each shard the locations are also random, however those random locations are read in order
    random.shuffle(domain_sampling_shard_list)
    
    os.makedirs(output_dir, exist_ok=True)
    shard_writer = ShardWriter(os.path.join(output_dir, "shard-%07d.tar"), maxcount=shard_size)

    chunk = []
    domain_counter_dict = defaultdict(int)
    for domain, shard_path in domain_sampling_shard_list:
        # get within shard inde (increasing oder)
        within_shard_idx = shard_sampling_locations_dict[shard_path].pop(0)
        # sample idx from shard. Due to sequential read, we can do this really fast
        data = data_readers[shard_path](within_shard_idx)
        if data is None:
            continue
        chunk.append(data)

        domain_counter_dict[domain] += 1
        if len(chunk) == shard_size:
            write_to_shard(chunk, shard_writer)
            chunk = []

    #! ignore last chunk to guarantee same size chunks in pipeline 
    shard_writer.close()

    # stats
    total_number_of_tokens = sum(domain_counter_dict.values()) * chunk_size
    total_time = time.time() - start_time
    logger.info(f"Finished! Processed {total_number_of_tokens} tokens in {total_time}. Per domain stats:")
    
    true_mixing_weights = {}
    for domain, count in domain_counter_dict.items():
        domain_tokens = count * chunk_size
        domain_ratio = domain_tokens / total_number_of_tokens
        domain_percentage = domain_ratio * 100
        true_mixing_weights[domain] = domain_ratio
        logger.info(f"{domain}: {domain_tokens} tokens, i.e. {domain_percentage:.2f}% of total tokens")
    
    # reorder to have mixing weights according to domains
    out_mixing_weights = []
    for domain in domains:
        out_mixing_weights.append(true_mixing_weights.get(domain, 0))
    
    if seed:
        logger.info(f"Resetting seed to original state {original_random_state}")
        random.setstate(original_random_state)
    
    return out_mixing_weights

    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--domains", type=str, nargs="+", required=True, help="Domains to be mixed")
    parser.add_argument("--manifests", type=str, nargs="+", required=True, help="Manifest of all datasets to be mixed")
    parser.add_argument("--mixing-weights", type=float, nargs="+", required=True, help="Weights of each dataset in the mixing")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    parser.add_argument("--token-count", type=int,  required=True, help="Number of output tokens")
    parser.add_argument("--chunk-size", type=int, default=2048, help="Chunk size, i.e., number of tokens in each line of tokenized data")
    parser.add_argument("--no-readers", type=int, default=1, help="Number of workers for parallel processing")

    args = parser.parse_args()

    mix_tokenized_data(args.domains, args.manifests, args.mixing_weights, args.output_dir, args.token_count, args.chunk_size, args.no_readers)
