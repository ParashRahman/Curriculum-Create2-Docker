# Copyright (c) 2018, The SenseAct Authors.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import os
import builtins
import tempfile, zipfile
import numpy as np


def create_callback(shared_returns, load_model_path=None):
    builtins.shared_returns = shared_returns
    builtins.load_model_path = load_model_path

    def kindred_callback(locals, globals):
        import tensorflow as tf
        saver = tf.train.Saver()

        shared_returns = globals['__builtins__']['shared_returns']
        if locals['iters_so_far'] == 0:
            path = globals['__builtins__']['load_model_path']
            if path is not None:
                # tf.reset_default_graph()
                # saver = tf.train.import_meta_graph(path + '.meta')
                saver.restore(tf.get_default_session(), path)
                # tf_load_session_from_pickled_model(globals['__builtins__']['load_model_data'])
                for i in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
                    print(i.eval())
        else:
            ep_rets = locals['seg']['ep_rets']
            ep_lens = locals['seg']['ep_lens']
            ep_ss = locals['seg']['ep_ss']
            if len(ep_rets):
                if not shared_returns is None:
                    shared_returns['write_lock'] = True
                    shared_returns['episodic_returns'] += ep_rets
                    shared_returns['episodic_lengths'] += ep_lens
                    shared_returns['episodic_ss'] += ep_ss
                    shared_returns['write_lock'] = False
                    np.save('ep_lens',
                            np.array(shared_returns['episodic_lengths']))
                    np.save('ep_rets',
                            np.array(shared_returns['episodic_returns']))
                    np.save('ep_ss',
                            np.array(shared_returns['episodic_ss']))
        fname = 'saved/model' + str(locals['iters_so_far']) + '.ckpt'
        saver.save(tf.get_default_session(), fname)
    return kindred_callback
