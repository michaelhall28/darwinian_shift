import pandas as pd
import numpy as np
import urllib.parse
import urllib.request
import os
import xmlschema


class UniprotLookupError(ValueError): pass


class UniprotLookup:
    #  The score of a mutation is based on which uniprot features it is part of
    # Each feature type can be given a different score
    # Any residues in multiple features will be placed in the one with the highest score.
    # This code can also be used to annotate mutations with the uniprot features.

    # Features generally have a single residue, or include a set of residues between the begin and end positions
    # However, disulfide bonds have a begin and end position but do not include the residues in between
    SPLIT_FEATURES = {'disulfide bond'}
    MATCH_VARIANT_FEATURES = {'sequence variant'}

    REQUIRED_COLS = ('position', 'begin_position', 'end_position', 'description')

    def __init__(self, uniprot_directory=None, store_xml=False, feature_types=None,
                 description_contains=None, uniprot_upload_lists='https://www.uniprot.org/uploadlists/',
                 uniprot_xml_url="https://www.uniprot.org/uniprot/{}.xml",
                 schema_location='https://www.uniprot.org/docs/uniprot.xsd', transcript_uniprot_mapping=None,
                 force_download=False, match_variant_change=True, name='Uniprot', ):
        """

        :param uniprot_directory:
        :param store_xml:
        :param feature_types: List of the types to test. They include 'signal peptide', 'chain', 'topological domain',
         'transmembrane region', 'domain', 'repeat', 'region of interest', 'metal ion-binding site', 'site',
         'modified residue', 'glycosylation site', 'disulfide bond', 'cross-link', 'splice variant', 'mutagenesis site',
         'sequence conflict', 'helix', 'strand'.
        :param description_contains: Text that the description must contain. E.g. you might want only topological domains
        with "Cytoplasmic" in the description
        :param uniprot_upload_lists:
        :param uniprot_xml_url:
        :param schema_location:
        :param transcript_uniprot_mapping: Sometimes uniprot may not know which entry matches the transcript, or will
        not return the desired match. For these cases, you can provide a dictionary like {ENST00000123456: P12345},
        or a file with lines like "ENST00000123456 P12345".
        A file in that format can be downloaded from ensembl biomart by selection the "Transcript stable ID" and
        "UniProtKB/Swiss-Prot ID". This can be useful for GRCh37.
        Anything transcripts not in the given file/dictionary will be matched using the uniprot mapping as usual.
        :param force_download:
        """
        self.uniprot_upload_lists = uniprot_upload_lists
        self.uniprot_xml_url = uniprot_xml_url
        self.schema = xmlschema.XMLSchema(schema_location)
        self.transcript_uniprot_mapping = None
        if isinstance(transcript_uniprot_mapping, dict):
            self.transcript_uniprot_mapping = transcript_uniprot_mapping
        elif isinstance(transcript_uniprot_mapping, str):
            self.transcript_uniprot_mapping = {}
            with open(transcript_uniprot_mapping) as fh:
                for line in fh:
                    try:
                        transcript, uniprot = line.strip().split()
                    except ValueError as e:
                        # Skip any rows with wrong number of entries (e.g. where uniprot id is missing).
                        continue
                    self.transcript_uniprot_mapping[transcript] = uniprot
        elif transcript_uniprot_mapping is not None:
            raise UniprotLookupError('transcript_uniprot_mapping must be dictionary or file path')


        self.uniprot_directory = uniprot_directory
        self.store_xml = store_xml
        self.force_download = force_download
        if feature_types is not None:
            if isinstance(feature_types, str):
                self.feature_types = [feature_types]
            else:
                self.feature_types = feature_types
        else:
            self.feature_types = 'all'

        if description_contains is not None:
            if isinstance(description_contains, str):
                description_contains = [description_contains]
            if len(description_contains) != len(self.feature_types):
                raise ValueError("Must be same number of description matches as there are features")
            self.description_contains = description_contains
        else:
            self.description_contains = None

        self.match_variant_change = match_variant_change  # Set to false to just match position of sequence variants

        self.name = name  # Will appear on some plot axes

    def __call__(self, seq_object):
        return self._get_scores(seq_object.null_mutations, seq_object.transcript_id)

    def get_uniprot_xml_from_acc(self, acc):
        url = self.uniprot_xml_url.format(acc)
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req) as f:
            response = f.read()
        return response.decode('utf-8')

    def get_uniprot_xml_from_transcript_id(self, transcript_id):
        params = {
            'from': 'ENSEMBL_TRS_ID',
            'to': 'ACC',
            'format': 'xml',
            'query': transcript_id
        }

        data = urllib.parse.urlencode(params)
        data = data.encode('utf-8')
        req = urllib.request.Request(self.uniprot_upload_lists, data)
        with urllib.request.urlopen(req) as f:
            response = f.read()
        return response.decode('utf-8')

    def get_pdb_structures_for_transcript(self, transcript_id):
        xml_dict = self.get_uniprot_xml_dict(transcript_id)
        return self._get_pdb_structures_from_xml_dict(xml_dict)

    def _get_pdb_structures_from_xml_dict(self, xml_dict):
        """
        Returns a list of PDB IDs.
        :param xml_dict:
        :return:
        """
        try:
            pdb_structures = []
            for p in xml_dict['entry'][0]['dbReference']:
                if p['@type'] == 'PDB':
                    d = {'pdb_id': p['@id']}
                    for property in p['property']:
                        if property['@type'] == 'chains':
                            d['chains'] = property['@value'].split("=")[0]  # Ignore the residues here, can use SIFTS
                        else:
                            d[property['@type']] = property['@value']
                    pdb_structures.append(d)

        except KeyError as e:
            # No features for this uniprot gene
            return UniprotLookupError('No pdb structures found in xml_dict')
        pdb_structures = pd.DataFrame(pdb_structures)
        if 'resolution' in pdb_structures.columns:
            pdb_structures['resolution'] = pdb_structures['resolution'].astype(float)
        return pdb_structures

    def get_features_from_xml(self, xml_dict):
        try:
            features = xml_dict['entry'][0]['feature']
        except KeyError as e:
            # No features for this uniprot gene
            return UniprotLookupError('No features found in xml_dict')

        # Need to unpack the feature dictionaries and make a pandas dataframe
        new_dicts = []
        for f in features:
            d = {}
            for k, v in f.items():
                if k.startswith("@"):
                    d[k[1:]] = v
                elif k == 'location':
                    # This is a dictionary with more dictionaries as the values
                    for k2, v2 in v.items():
                        if k2 == 'position':
                            d['position'] = v2['@position']
                            d['position_status'] = v2['@status']
                        elif k2 == '@sequence':
                            d['location_sequence'] = v2
                        else:
                            # try:
                            for k3, v3 in v2.items():
                                d[k2 + '_' + k3[1:]] = v3
                            # except AttributeError as e:
                            #     print(v2)
                                # raise e
                else:
                    d[k] = v

            new_dicts.append(d)

        d = pd.DataFrame(new_dicts)
        for col in self.REQUIRED_COLS:
            if col not in d.columns:
                d[col] = np.nan

        return d

    def get_uniprot_xml_dict(self, transcript_id):
        xml_dict = None
        uniprot_id = None
        if self.transcript_uniprot_mapping is not None:
            uniprot_id = self.transcript_uniprot_mapping.get(transcript_id)

        if self.uniprot_directory is not None and not self.force_download:  # Look for an already downloaded version of the uniprot entry
            if uniprot_id is not None:
                # First look for a file with the given accession number
                xml_path = os.path.join(self.uniprot_directory, transcript_id + '_' + uniprot_id + ".xml")
                if os.path.exists(xml_path):
                    xml_dict = self.schema.to_dict(xml_path)

            if xml_dict is None:
                xml_path = os.path.join(self.uniprot_directory, transcript_id + ".xml")
                if os.path.exists(xml_path):
                    xml_dict = self.schema.to_dict(xml_path)

                if xml_dict is not None and uniprot_id is not None:
                    # Check if the existing file matches the given accession id
                    accessions = xml_dict['entry'][0]['accession']
                    if uniprot_id not in accessions:
                        # Need to download the requested accession
                        xml = self.get_uniprot_xml_from_acc(uniprot_id)
                        xml_dict = self.schema.to_dict(xml)
                        if self.store_xml and self.uniprot_directory is not None:
                            with open(os.path.join(self.uniprot_directory,
                                                   transcript_id + '_' + uniprot_id + ".xml"), 'w') as fh:
                                fh.writelines(xml)

        if xml_dict is None:  #  No version already downloaded or forcing new download. Get the data from the uniprot api
            if uniprot_id is not None:
                # Get the requested uniprot entry
                xml = self.get_uniprot_xml_from_acc(uniprot_id)
                if len(xml) == 0:
                    # Failed to find the uniprot data
                    raise UniprotLookupError('Failed to download uniprot xml with accession id {}'.format(uniprot_id))
            else:
                # Use uniprot to find the correct entry using the transcript id
                xml = self.get_uniprot_xml_from_transcript_id(transcript_id)
                if len(xml) == 0:
                    # Failed to find the uniprot data
                    raise UniprotLookupError('Failed to find and download uniprot xml for transcript {}'.format(
                        transcript_id))

            xml_dict = self.schema.to_dict(xml)
            if self.store_xml and self.uniprot_directory is not None:
                if uniprot_id is not None:
                    xml_path = os.path.join(self.uniprot_directory,
                                            transcript_id + '_' + uniprot_id + ".xml")
                else:
                    xml_path = os.path.join(self.uniprot_directory, transcript_id + ".xml")
                with open(xml_path, 'w') as fh:
                    fh.writelines(xml)

        return xml_dict

    def get_uniprot_data(self, transcript_id):
        xml_dict = self.get_uniprot_xml_dict(transcript_id)

        return self.get_features_from_xml(xml_dict)

    def _add_annotation(self, value, new_text, sep):
        if pd.isnull(new_text):
            new_text = 'TRUE'
        if pd.isnull(value):
            return new_text
        else:
            return value + sep + new_text

    def annotate_dataframe(self, df, transcript_id, sep='|||'):
        feature_columns = []  # List the columns used for the annotating of the mutations
        transcript_features = self.get_uniprot_data(transcript_id)
        if 'location_sequence' in transcript_features.columns:
            # Entry defined on an alternative isoform. Remove here as it could match to the wrong residues.
            transcript_features = transcript_features[pd.isnull(transcript_features['location_sequence'])]
        if transcript_features is not None:
            if self.feature_types == 'all':
                transcript_feature_types = transcript_features['type'].unique()
            else:
                transcript_feature_types = self.feature_types
            if self.description_contains is None:
                transcript_desc_contains = [None] * len(transcript_feature_types)
            else:
                transcript_desc_contains = self.description_contains

            df['score'] = np.nan
            for f, d in zip(transcript_feature_types, transcript_desc_contains):
                if d is None or d == "":
                    col = f
                    feature_rows = transcript_features[transcript_features['type'] == f]
                else:
                    col = f + '_' + d
                    feature_rows = transcript_features[(transcript_features['type'] == f) &
                                                   transcript_features['description'].str.contains(d)]

                df[col] = np.nan
                feature_columns.append(col)

                if f in self.SPLIT_FEATURES:  # Just matching the start and end position, not any residues in between
                    for i, row in feature_rows.iterrows():
                        df.loc[df['residue'].isin([row['begin_position'], row['end_position']]), 'score'] = 1
                        df.loc[df['residue'].isin([row['begin_position'], row['end_position']]), col] = \
                            df.loc[df['residue'].isin([row['begin_position'], row['end_position']]), col].apply(
                                lambda x: self._add_annotation(x,
                                                               row['description'],
                                                               sep))
                elif f in self.MATCH_VARIANT_FEATURES and self.match_variant_change:  # Also need to match the amino acid change of the variant
                    for i, row in feature_rows.iterrows():
                        pos = row['position']
                        ref_aa = row['original']
                        alt_aa = row['variation']
                        if not pd.isnull(alt_aa):
                            if len(alt_aa) > 1:
                                print(alt_aa)
                                print(len(alt_aa))
                                print(type(alt_aa))
                                print(row)
                            for aa in alt_aa:
                                df.loc[(df['residue'] == pos) & (df['aaref'] == ref_aa) &
                                       (df['aamut'] == aa), 'score'] = 1
                                df.loc[(df['residue'] == pos) & (df['aaref'] == ref_aa) &
                                       (df['aamut'] == aa), col] = df.loc[(df['residue'] == pos) & (df['aaref'] == ref_aa) &
                                       (df['aamut'] == aa), col].apply(lambda x: self._add_annotation(x, row['description'],
                                                                                                             sep))
                else:
                    for i, row in feature_rows.iterrows():
                        if pd.isnull(row['position']):  # Region of protein
                            start = row['begin_position']
                            end = row['end_position']
                        else:  # Single residue
                            start = row['position']
                            end = row['position']
                        df.loc[(df['residue'] >= start) & (df['residue'] <= end), 'score'] = 1
                        df.loc[(df['residue'] >= start) & (df['residue'] <= end), col] = \
                            df.loc[(df['residue'] >= start) &
                                   (df['residue'] <= end), col].apply(lambda x: self._add_annotation(x,
                                                                                                     row[
                                                                                                         'description'],
                                                                                                     sep))

        return df, feature_columns

    def _get_scores(self, df, transcript_id):
        try:
            df, _ = self.annotate_dataframe(df, transcript_id)
        except UniprotLookupError as e:
            print(type(e).__name__, e)
            return None

        return df['score'].fillna(value=0).values
